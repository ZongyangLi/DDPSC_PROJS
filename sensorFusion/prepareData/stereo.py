'''
Created on Oct 7, 2019

@author: zli
'''

import sys, os.path, json, random, argparse
from glob import glob
from os.path import join
import numpy as np
from scipy.ndimage.filters import convolve
from PIL import Image
from datetime import date, timedelta,datetime
import shutil
import multiprocessing
import cv2
import terra_common
from networkx.algorithms.distance_measures import center
from pip._vendor.urllib3.util.ssl_ import PROTOCOL_TLS

fov_adj = 0.97

# Test by Baker
FOV_MAGIC_NUMBER = 0.1552 
FOV_IN_2_METER_PARAM = 0.837 # since we don't have a true value of field of view in 2 meters, we use this parameter(meter) to estimate fov in Y-
SEASON_10_SHIFT = -0.1
scan_shift = -0.03

os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'

def options():
    
    parser = argparse.ArgumentParser(description='RGB Cropping',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-i", "--in_dir", help="raw rgb directory")
    parser.add_argument("-o", "--out_dir", help="output rgb image directory")
    parser.add_argument("-p", "--plot_dir", help="output cropped plot directory")
    
    args = parser.parse_args()

    return args


def full_day_multi_process(in_dir, out_path, plot_dir, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = [os.path.join(in_dir,o) for o in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,o))]
    out_dirs = [os.path.join(out_path,o) for o in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,o))]
    numDirs = len(list_dirs)
    
    print ("Starting bin to png conversion...")
    pool = multiprocessing.Pool()
    NUM_THREADS = min(4,numDirs)
    print('numDirs:{}   NUM_THREADS:{}'.format(numDirs, NUM_THREADS))
    for cpu in range(NUM_THREADS):
        pool.apply_async(bin_to_png, [list_dirs[cpu::NUM_THREADS], out_dirs[cpu::NUM_THREADS], plot_dir, convt])
    pool.close()
    pool.join()
    print ("Completed bin to png conversion...")
    
    return

def bin_to_png(in_dirs, out_dirs, plot_dir, convt):
    for i, o in zip(in_dirs, out_dirs):
        #Generate jpgs and geoTIFs from .bin
        try:
            singe_image_process(i, o, plot_dir, convt)
            #bin_to_geotiff.stereo_test(s, s)
        except Exception as ex:
            fail("\tFailed to process folder %s: %s" % (i, str(ex)))

def fail(reason):
    print (reason)

def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict

def find_input_files(in_dir):
    metadata_suffix = '_metadata.json'
    metas = [os.path.basename(meta) for meta in glob(join(in_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        fail('No metadata file found in input directory.')
        return None, None, None

    guids = [meta[:-len(metadata_suffix)] for meta in metas]
    ims_left = [guid + '_left.bin' for guid in guids]
    ims_right = [guid + '_right.bin' for guid in guids]

    return metas, ims_left, ims_right

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))

def get_image_shape(metadata, which):
    try:
        im_meta = metadata['lemnatec_measurement_metadata']['sensor_variable_metadata']
        fmt = im_meta['image format %s image' % which]
        if fmt != 'BayerGR8':
            fail('Unknown image format: ' + fmt)
        width = im_meta['width %s image [pixel]' % which]
        height = im_meta['height %s image [pixel]' % which]
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        width = int(width)
        height = int(height)
    except ValueError as err:
        fail('Corrupt image dimension, ' + err.args[0])
    return (width, height)

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        cam_x = cam_meta["location in camera box x [m]"]
        cam_y = cam_meta["location in camera box y [m]"]

        
        if "location in camera box z [m]" in cam_meta: # this may not be in older data
            cam_z = cam_meta["location in camera box z [m]"]
        else:
            cam_z = 0
            
        y_ends = gantry_meta['y end pos [m]']

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        return None, None
        
    position = [float(gantry_x), float(gantry_y), float(gantry_z)]
    center_position = [position[0]+float(cam_x), position[1]+float(cam_y), position[2]+float(cam_z)]
    
    return center_position, y_ends

def get_fov(metadata, camHeight):
    try:
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        fov = cam_meta["field of view at 2m in x- y- direction [m]"]
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        fov_list = fov.replace("[","").replace("]","").split()
        fov_x = 1.857#float(fov_list[0])
        fov_y = 1.246#float(fov_list[1])    # fov in metadata is not consistance

        # given fov is at 2m, so need to convert for actual height
        fov_x = (camHeight * (fov_x))/2
        fov_y = (camHeight * (fov_y))/2

    except ValueError as err:
        fail('Corrupt FOV inputs, ' + err.args[0])
    return [fov_x, fov_y]


def get_fov_formular(metadata, camHeight):
    
    fov_x = 1.857
    fov_y = 1.246
    
    shape = [3296,2472]
    
    # test by Baker
    gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
    gantry_z = gantry_meta["position z [m]"]
    fov_offset = (float(gantry_z) - 2) * FOV_MAGIC_NUMBER
    fov_y = fov_y*(FOV_IN_2_METER_PARAM + fov_offset)
    fov_x = (fov_y)/shape[1]*shape[0]
    
    return [fov_x, fov_y]

def demosaic(im):
    # Assuming GBRG ordering.
    B = np.zeros_like(im)
    R = np.zeros_like(im)
    G = np.zeros_like(im)
    R[0::2, 1::2] = im[0::2, 1::2]
    B[1::2, 0::2] = im[1::2, 0::2]
    G[0::2, 0::2] = im[0::2, 0::2]
    G[1::2, 1::2] = im[1::2, 1::2]

    fG = np.asarray(
            [[0, 1, 0],
             [1, 4, 1],
             [0, 1, 0]]) / 4.0
    fRB = np.asarray(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]) / 4.0

    im_color = np.zeros(im.shape+(3,), dtype='uint8') #RGB
    im_color[:, :, 0] = convolve(R, fRB)
    im_color[:, :, 1] = convolve(G, fG)
    im_color[:, :, 2] = convolve(B, fRB)
    return im_color

def crop_rgb_imageToPlot(in_dir, out_dir, plot_dir, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    
    list_dirs = os.listdir(in_dir)
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        out_path = os.path.join(out_dir, d)
        
        if not os.path.isdir(in_path):
            continue
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
            
        try:
            singe_image_process(in_path, out_path, plot_dir, convt)
        except ValueError as err:
            fail('Error cropping file:' + in_path)
            continue

    
    return

def metadata_to_boundsList(center_position, fov, image_shape, y_ends, convt):
    
    field_dist_per_pix = fov[1]/image_shape[1]
    
    # A: right lower point in the image
    x_a = center_position[0]-fov[0]/2
    y_a = center_position[1]-fov[1]/2
    
    # B: left upper point in the image
    x_b = center_position[0]+fov[0]/2
    y_b = center_position[1]+fov[1]/2
    
    img_x_index = np.zeros(image_shape[1])
    img_y_index = np.zeros(image_shape[0])
    
    for i in range(image_shape[1]):
        field_y_coordinate = y_b - field_dist_per_pix*i
        field_col = convt.fieldPosition_to_fieldPartition(x_a, field_y_coordinate)[1]
        img_x_index[i] = field_col
        
    for j in range(image_shape[0]):
        field_x_coordinate = x_b - field_dist_per_pix*j
        field_range = convt.fieldPosition_to_fieldPartition(field_x_coordinate, y_a)[0]
        img_y_index[j] = field_range
        
    img_x_index.astype('uint16')
    img_y_index.astype('uint16')
    
    col_unique = np.unique(img_x_index)
    range_unique = np.unique(img_y_index)
    
    col_list = []
    range_list = []
    
    for i in range(len(col_unique)):
        col_num = col_unique[i]
        if col_num == 0:
            continue
        target_indexs = np.where(img_x_index==col_num)
        min_x = target_indexs[0].min()
        max_x = target_indexs[0].max()
        col_list.append([col_num, min_x, max_x])
        
    for j in range(len(range_unique)):
        range_num = range_unique[j]
        if range_num == 0:
            continue
        target_indexs = np.where(img_y_index==range_num)
        min_x = target_indexs[0].min()
        max_x = target_indexs[0].max()
        range_list.append([range_num, min_x, max_x])
        
    rel_list = []
    for col_item in col_list:
        for range_item in range_list:
            range_num = int(range_item[0])
            col_num = int(col_item[0])
            i_x_min = col_item[1]
            i_x_max = col_item[2]
            i_y_min = range_item[1]
            i_y_max = range_item[2]
            image_roiBox = [i_x_min, i_x_max, i_y_min, i_y_max]
            
            f_x_min = x_a+(image_shape[0]-i_y_max)*field_dist_per_pix
            f_x_max = x_a+(image_shape[0]-i_y_min)*field_dist_per_pix
            f_y_min = y_a+(image_shape[1]-i_x_max)*field_dist_per_pix
            f_y_max = y_a+(image_shape[1]-i_x_min)*field_dist_per_pix
            field_roiBox = [f_x_min,f_x_max,f_y_min,f_y_max]
                
            # add scan shift to y axis
            #field_roiBox = add_scan_shift_to_field_roiBox(field_roiBox, y_ends)
            
            # remove roi which is too small
            if check_if_roi_too_small(field_roiBox):
                continue
            
            rel_list.append([range_num, col_num, image_roiBox, field_roiBox])
    
    return rel_list

x_range_length_threshold = 0.4
y_column_length_thershold = 0.2

def check_if_roi_too_small(field_roiBox):
    
    if field_roiBox[1] - field_roiBox[0] < x_range_length_threshold:
        return True
    if field_roiBox[3] - field_roiBox[2] < y_column_length_thershold:
        return True

    return False

def metadata_to_imageBoundaries(center_position, fov, image_shape, convt):
    
    plotNum = 0
    roiBox = []
    
    field_dist_per_pix = fov[1]/image_shape[1]
    
    # A: right lower point in the image
    x_a = center_position[0]-fov[0]/2
    y_a = center_position[1]-fov[1]/2
    
    # B: left upper point in the image
    x_b = center_position[0]+fov[0]/2
    y_b = center_position[1]+fov[1]/2
    
    row_a, col_a = convt.fieldPosition_to_fieldPartition(x_a, y_a)   # A range/column
    row_b, col_b = convt.fieldPosition_to_fieldPartition(x_b, y_b)   # B range/column
    if 0 in [row_a, row_b, col_a, col_b]:
        print('row in the gap')
        return None, None, None
    
    if row_a == row_b and col_a == col_b:
        plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
        roiBox = [0,image_shape[0],0,image_shape[1]]
        
    
    if row_a == row_b and col_a != col_b:
        xd_0 = (y_b-convt.np_bounds[row_a-1][col_b-1][2])/fov[1]*image_shape[1]
        #xd_1 = (convt.np_bounds[row_a-1][col_a-1][3]-y_a)/fov[1]*image_shape[1]
        
        if xd_0 > image_shape[1]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, image_shape[0], 0, xd_0]
        else:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [0, image_shape[0],xd_0,image_shape[1]]
            
    if row_a != row_b and col_a == col_b:
        yd_0 = (x_b-convt.np_bounds[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if yd_0 > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, yd_0, 0, image_shape[1]]
        else:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [yd_0, image_shape[0],0,image_shape[1]]
            
    if row_a != row_b and col_a != col_b:
        x_m = (y_b-convt.np_bounds[row_a-1][col_b-1][2])/fov[1]*image_shape[1]
        y_m = (x_b-convt.np_bounds[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if x_m > image_shape[1]/2 and y_m > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, y_m, 0, x_m]
        elif x_m > image_shape[1]/2 and y_m < image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_b)
            roiBox = [y_m, image_shape[0], 0, x_m]
        elif x_m <= image_shape[1]/2 and y_m <= image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [y_m, image_shape[0], x_m, image_shape[1]]
        elif x_m < image_shape[1]/2 and y_m > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_a)
            roiBox = [0, y_m, x_m, image_shape[1]]
            
    xmin = x_a+(image_shape[0]-roiBox[1])*field_dist_per_pix
    xmax = x_a+(image_shape[0]-roiBox[0])*field_dist_per_pix
    ymin = y_a+(image_shape[1]-roiBox[3])*field_dist_per_pix
    ymax = y_a+(image_shape[1]-roiBox[2])*field_dist_per_pix
    field_roiBox = [xmin,xmax,ymin,ymax]
    
    roiBox = [int(i) for i in roiBox]
    
    return plotNum, roiBox, field_roiBox

def metadata_to_imageBoundaries_with_gaps(center_position, fov, image_shape, convt):
    
    plotNum = 0
    roiBox = []
    
    field_dist_per_pix = fov[1]/image_shape[1]
    
    # A: right lower point in the image
    x_a = center_position[0]-fov[0]/2
    y_a = center_position[1]-fov[1]/2
    
    # B: left upper point in the image
    x_b = center_position[0]+fov[0]/2
    y_b = center_position[1]+fov[1]/2
    
    row_a, col_a = convt.fieldPosition_to_fieldPartition_w_gaps(x_a, y_a)   # A range/column
    row_b, col_b = convt.fieldPosition_to_fieldPartition_w_gaps(x_b, y_b)   # B range/column
    if 0 in [row_a, row_b, col_a, col_b]:
        return None, None, None
    
    if row_a == row_b and col_a == col_b:
        if row_a%2 == 0:
            return None, None, None
        plot_row = row_a//2
        plotNum = convt.fieldPartition_to_plotNum(plot_row, col_a)
        roiBox = [0,image_shape[0],0,image_shape[1]]
        
    
    if row_a == row_b and col_a != col_b:
        if row_a%2 == 0:
            return None, None, None
        plot_row = row_a//2
        
        xd_0 = (y_b-convt.np_bounds[plot_row-1][col_b-1][2])/fov[1]*image_shape[1]
        #xd_1 = (convt.np_bounds[row_a-1][col_a-1][3]-y_a)/fov[1]*image_shape[1]
        
        if xd_0 > image_shape[1]/2:
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_b)
            roiBox = [0, image_shape[0], 0, xd_0]
        else:
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_a)
            roiBox = [0, image_shape[0],xd_0,image_shape[1]]
            
    if row_a != row_b and col_a == col_b:
        yd_0 = (x_b-convt.np_bounds_w_gaps[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if yd_0 > image_shape[0]/2:
            if row_b%2 == 0:
                return None, None, None
            plot_row = row_b//2
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_b)
            roiBox = [0, yd_0, 0, image_shape[1]]
        else:
            if row_a%2 == 0:
                return None, None, None
            plot_row = row_a//2
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_a)
            roiBox = [yd_0, image_shape[0],0,image_shape[1]]
            
    if row_a != row_b and col_a != col_b:
        x_m = (y_b-convt.np_bounds_w_gaps[row_a-1][col_b-1][2])/fov[1]*image_shape[1]
        y_m = (x_b-convt.np_bounds_w_gaps[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if x_m > image_shape[1]/2 and y_m > image_shape[0]/2:
            if row_b%2 == 0:
                return None, None, None
            plot_row = row_b//2            
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_b)
            roiBox = [0, y_m, 0, x_m]
        elif x_m > image_shape[1]/2 and y_m < image_shape[0]/2:
            if row_a%2 == 0:
                return None, None, None
            plot_row = row_a//2
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_b)
            roiBox = [y_m, image_shape[0], 0, x_m]
        elif x_m <= image_shape[1]/2 and y_m <= image_shape[0]/2:
            if row_a%2 == 0:
                return None, None, None
            plot_row = row_a//2
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_a)
            roiBox = [y_m, image_shape[0], x_m, image_shape[1]]
        elif x_m < image_shape[1]/2 and y_m > image_shape[0]/2:
            if row_b%2 == 0:
                return None, None, None
            plot_row = row_b//2
            plotNum = convt.fieldPartition_to_plotNum(plot_row, col_a)
            roiBox = [0, y_m, x_m, image_shape[1]]
            
    xmin = x_a+(image_shape[0]-roiBox[1])*field_dist_per_pix
    xmax = x_a+(image_shape[0]-roiBox[0])*field_dist_per_pix
    ymin = y_a+(image_shape[1]-roiBox[3])*field_dist_per_pix
    ymax = y_a+(image_shape[1]-roiBox[2])*field_dist_per_pix
    field_roiBox = [xmin,xmax,ymin,ymax]
    
    roiBox = [int(i) for i in roiBox]
    
    return plotNum, roiBox, field_roiBox

def add_season10_shift(center_position, shift_meter):
    
    center_position[1] = center_position[1] + shift_meter
    
    return center_position

def add_scan_shift(center_position, y_ends):
    
    if y_ends == '0':     # + shift
        center_position[1] = center_position[1]+scan_shift
    else:               # - shift   
        center_position[1] = center_position[1]-scan_shift
        
    return center_position
    

def add_scan_shift_to_field_roiBox(field_roiBox, y_ends):
    
    if y_ends == '0':     # + shift
        field_roiBox[2] = field_roiBox[2]+scan_shift
        field_roiBox[3] = field_roiBox[3]+scan_shift
    else:               # - shift   
        field_roiBox[2] = field_roiBox[2]-scan_shift
        field_roiBox[3] = field_roiBox[3]-scan_shift
        
    return field_roiBox

def singe_image_process_old_version(in_dir, out_dir, plot_dir, convt):
    
    time_stamp = os.path.basename(in_dir)
    print(time_stamp)
    
    # find input files
    metas, ims_left, ims_right = find_input_files(in_dir)
    if metas == None:
        print(in_dir)
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    base_name = os.path.basename(ims_left[0])[:-4]
    out_path = os.path.join(out_dir, '{}.png'.format(base_name))
    #if os.path.isfile(out_path):
    #    return
    
    # parse meta data
    metadata = lower_keys(load_json(os.path.join(in_dir, metas[0])))
    center_position, y_ends = parse_metadata(metadata)
    
    if center_position == None:
        print(in_dir)
        return
    fov = get_fov_formular(metadata, center_position[2])
    
    image_shape = (3296, 2472)
    
    # center position/fov/imgSize to plot number, image boundaries, only dominated plot for now
    try:
        plotNum, roiBox, field_roiBox = metadata_to_imageBoundaries(center_position, fov, image_shape, convt)
    except ValueError as err:
        fail('Error metadata_to_imageBoundaries:' + in_dir)
        return
            
    if plotNum == None:
        print(time_stamp+'plotNum could not found.\n')
        return
    plot_row, plot_col = convt.plotNum_to_fieldPartition(plotNum)
    if plotNum == 0:
        return
    
    # add scan shift to y axis
    field_roiBox = add_scan_shift_to_field_roiBox(field_roiBox, y_ends)
        
    # bin to image
    try:
        im = np.fromfile(join(in_dir, ims_left[0]), dtype='uint8').reshape(image_shape[::-1])
    except ValueError as err:
        fail('Error bin to image: ' + err.args[0])
        print(in_dir)
        return
    
    im_color = demosaic(im)
    im_color = (np.rot90(im_color))

    Image.fromarray(im_color).save(out_path)

    
    # crop image
    cv_img = im_color
    roi_img = cv_img[roiBox[0]:roiBox[1], roiBox[2]:roiBox[3]]
    
    # save image
    save_dir = '{0:02d}-{1:02d}-{2:04d}'.format(plot_row, plot_col, plotNum)
    s_d = os.path.join(plot_dir, save_dir)
    if not os.path.isdir(s_d):
        os.mkdir(s_d)
        
    dst_img_path = os.path.join(s_d, '{}.png'.format(time_stamp))
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dst_img_path, roi_img)
    #Image.fromarray(roi_img).save(dst_img_path)
    
    # write new json file
    src_json_data = load_json(os.path.join(in_dir, metas[0]))
    
    add_metadata = {'gwu_added_metadata': {'xmin':field_roiBox[0], 'xmax':field_roiBox[1], 'ymin':field_roiBox[2], 'ymax':field_roiBox[3]}}
    dst_json_data = dict(src_json_data)
    dst_json_data.update(add_metadata)
    
    
    dst_json_path = os.path.join(s_d, '{}.json'.format(time_stamp))
    with open(dst_json_path, 'w') as outfile:
        json.dump(dst_json_data, outfile)
    
    return

def singe_image_process(in_dir, out_dir, plot_dir, convt):
    
    time_stamp = os.path.basename(in_dir)
    
    # find input files
    metas, ims_left, ims_right = find_input_files(in_dir)
    if metas == None:
        print(in_dir)
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    base_name = os.path.basename(ims_left[0])[:-4]
    out_path = os.path.join(out_dir, '{}.png'.format(base_name))
    #if os.path.isfile(out_path):
    #    return
    
    # parse meta data
    metadata = lower_keys(load_json(os.path.join(in_dir, metas[0])))
    center_position, y_ends = parse_metadata(metadata)
    
    if center_position == None or y_ends == None:
        print(in_dir)
        return
    
    # add season 10 shift to center position y
    if convt.seasonNum == 10:
        center_position = add_season10_shift(center_position, SEASON_10_SHIFT)
        
    # add scan shift to center position y
    center_position = add_scan_shift(center_position, y_ends)
    
    fov = get_fov_formular(metadata, center_position[2])
    
    image_shape = (3296, 2472)
    
    # center position/fov/imgSize to plot number, image boundaries, only dominated plot for now
    try:
        roi_list = metadata_to_boundsList(center_position, fov, image_shape, y_ends, convt)
    except ValueError as err:
        fail('Error metadata_to_imageBoundaries:' + in_dir)
        return
    
    # bin to image
    try:
        im = np.fromfile(join(in_dir, ims_left[0]), dtype='uint8').reshape(image_shape[::-1])
    except ValueError as err:
        fail('Error bin to image: ' + err.args[0])
        print(in_dir)
        return
    
    im_color = demosaic(im)
    im_color = (np.rot90(im_color))

    Image.fromarray(im_color).save(out_path)

    
    # crop image
    cv_img = im_color
    #im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    for roi_item in roi_list:
        plot_row, plot_col, roiBox, field_roiBox = roi_item
    
        roi_img = cv_img[roiBox[2]:roiBox[3], roiBox[0]:roiBox[1]]
        
        if roi_img.size ==0:
            continue
    
        # save image
        save_dir = '{0:02d}_{1:02d}'.format(plot_row, plot_col)
        s_d = os.path.join(plot_dir, save_dir)
        if not os.path.isdir(s_d):
            os.mkdir(s_d)
            
        dst_img_path = os.path.join(s_d, '{}.png'.format(time_stamp))
        save_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(dst_img_path, save_img)
    
        # write new json file
        src_json_data = load_json(os.path.join(in_dir, metas[0]))
        
        add_metadata = {'gwu_added_metadata': {'xmin':field_roiBox[0], 'xmax':field_roiBox[1], 'ymin':field_roiBox[2], 'ymax':field_roiBox[3]}}
        dst_json_data = dict(src_json_data)
        dst_json_data.update(add_metadata)
        
        
        dst_json_path = os.path.join(s_d, '{}.json'.format(time_stamp))
        with open(dst_json_path, 'w') as outfile:
            json.dump(dst_json_data, outfile)
    
    return


def extract_roiBox_from_metadata(metadata):
    
    gwu_meta = metadata['gwu_added_metadata']
    xmin = gwu_meta["xmin"]
    xmax = gwu_meta["xmax"]
    ymin = gwu_meta["ymin"]
    ymax = gwu_meta['ymax']
    
    return xmin,xmax,ymin,ymax

def stitch_plot_rgb_image(in_dir, out_dir, str_date, convt):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # get plot boundaries
    dir_name = os.path.basename(in_dir)
    plotId = dir_name.split('_')
    plot_row = int(plotId[0])
    plot_col = int(plotId[1])
    
    plot_bounds = convt.np_bounds[plot_row-1][plot_col-1]
       
    # loop image, patch it to target image
    metadata_suffix = '.json'
    metas = [os.path.basename(meta) for meta in glob(join(in_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return
    
    # init output image
    metadata = lower_keys(load_json(os.path.join(in_dir, metas[0])))
    center_position, y_ends = parse_metadata(metadata)
    if center_position == None:
        return
    fov = get_fov_formular(metadata, center_position[2])
    field_dist_per_pix = fov[0]/3296
    img_wids = int(round((plot_bounds[3]-plot_bounds[2])/field_dist_per_pix))+2000
    img_hts = int(round((plot_bounds[1]-plot_bounds[0])/field_dist_per_pix))+2000
    stitched_img = np.zeros((img_hts,img_wids,3),np.uint8)
    
    rect_x_min = img_wids
    rect_y_min = img_hts
    rect_x_max = 0
    rect_y_max = 0
    
    
    start_offset = 500
    for json_file in metas:
        json_path = os.path.join(in_dir, json_file)
        rgb_file = json_path[:-4]+'png'
        
        if not os.path.isfile(rgb_file):
            continue
        
        metadata = load_json(json_path)
        xmin,xmax,ymin,ymax = extract_roiBox_from_metadata(metadata)
        img = cv2.imread(rgb_file)
        height,width = img.shape[:2]
        x_start = int(round((plot_bounds[3]-ymax)/field_dist_per_pix))+start_offset
        y_start = int(round((plot_bounds[1]-xmax)/field_dist_per_pix))+start_offset
        if x_start < 0 or y_start < 0:
            print(1)
            continue
        
        if x_start < rect_x_min :
            rect_x_min = x_start
        if y_start < rect_y_min:
            rect_y_min = y_start
        if x_start+width > rect_x_max:
            rect_x_max = x_start+width
        if y_start+height > rect_y_max:
            rect_y_max = y_start+height
        
        stitched_img[y_start:y_start+height, x_start:x_start+width] = img
        
    if rect_y_max < rect_y_min or rect_x_max < rect_x_min:
        return
    
    save_img = stitched_img[rect_y_min:rect_y_max, rect_x_min:rect_x_max]
    
    # save output
    out_file_name = '{}_{}.png'.format(str_date, dir_name)
    
    cv2.imwrite(os.path.join(out_dir, out_file_name), save_img)
    
    return

def test():
    
    str_date = '2020-02-03'
    
    in_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/stereoTop/', str_date)
    out_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/StitchedPlotRGB', str_date)
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query(str_date)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    list_dirs = os.listdir(in_dir)
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        out_path = os.path.join(out_dir, d)
        
        if not os.path.isdir(in_path):
            continue
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        stitch_plot_rgb_image(in_path, out_path, str_date, convt)
    
    return

def test_single_dir():
    
    in_dir = '/media/zli/Elements/ua-mac/Level_2/RgbCropToPlot/2017-05-05/22-10-343'
    out_dir = '/media/zli/Elements/ua-mac/Level_2/StitchedPlotRGB/2017-05-05/22-10-343'
    str_date = '2018-05-24'
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query(str_date)
    
    stitch_plot_rgb_image(in_dir, out_dir, str_date, convt)
    
    stitch_plot_rgb_image('/media/zli/Elements/ua-mac/Level_2/RgbCropToPlot/2017-05-05/41-09-649', '/media/zli/Elements/ua-mac/Level_2/StitchedPlotRGB/2017-05-05/41-09-649', str_date, convt)
    
    return

def full_season_crop_rgb(raw_rgb_dir, out_dir, plot_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        raw_path = os.path.join(raw_rgb_dir, str_date)
        
        out_path = os.path.join(out_dir, str_date)
        
        plot_path = os.path.join(plot_dir, str_date)
        
        if not os.path.isdir(raw_path):
            continue
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
            
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        
        #crop_rgb_imageToPlot(raw_path, out_path, plot_path, convt)
        full_day_multi_process(raw_path, out_path, plot_path, convt)
    
    return

def full_season_rgb_stitch(png_dir, stitch_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    print(deltaDay.days)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        png_path = os.path.join(png_dir, str_date)
        
        stitch_path = os.path.join(stitch_dir, str_date)
        
        if not os.path.isdir(png_path):
            continue
        
        if not os.path.isdir(stitch_path):
            os.makedirs(stitch_path)
            
        list_dirs = os.listdir(png_path)
    
        for d in list_dirs:
            in_path = os.path.join(png_path, d)
            out_path = os.path.join(stitch_path, d)
            
            if not os.path.isdir(in_path):
                continue
            
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
        
            stitch_plot_rgb_image(in_path, out_path, str_date, convt)
    
    return

def main():
    start_date = '2020-02-03'  # S9 start date
    end_date = '2020-02-03'   # S9 end date
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2020-02-03') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    #args = options()
    
    raw_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/stereoTop'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_1/stereoTop'
    plot_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/rgb_crop'
    stitch_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/StitchedPlotRGB'
    
    full_season_crop_rgb(raw_dir, out_dir, plot_dir, start_date, end_date, convt)
    
    full_season_rgb_stitch(plot_dir, stitch_dir, start_date, end_date, convt)


if __name__ == '__main__':
    
    main()
    #test()
    #test_single_dir()
    
    
    
    
    
    
    
    