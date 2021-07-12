'''
Created on Apr 17, 2020

@author: zli

Given a RGB crop image file, its metadata, a txt file contain single plant bounding boxes in the RGB file, try to crop the same area in other sensor, other days.
'''

import sys, os.path, json, random, argparse
import gzip
import shutil
from glob import glob
from os.path import join
import numpy as np
from datetime import date, timedelta,datetime
import multiprocessing

import cv2
import terra_common
#from _operator import sub

os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'

THERMAL_SHIFT_X = 50
THERMAL_SHIFT_Y = 80

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        print('Corrupt metadata file, ' + str(ex))
        return
    
def extract_roiBox_from_metadata(metadata):
    
    gwu_meta = metadata['gwu_added_metadata']
    xmin = gwu_meta["xmin"]
    xmax = gwu_meta["xmax"]
    ymin = gwu_meta["ymin"]
    ymax = gwu_meta['ymax']
    
    return [xmin,xmax,ymin,ymax]

def extract_roiBox_from_metadata_laser(metadata):
    
    gwu_meta = metadata['cropping_variable_metadata']
    field_bbox_mm = gwu_meta['field_bbox_mm']
    xmin = field_bbox_mm['x']["min"]
    xmax = field_bbox_mm['x']["max"]
    ymin = field_bbox_mm['y']["min"]
    ymax = field_bbox_mm['y']['max']
    
    xmin = float(xmin)/1000
    xmax = float(xmax)/1000
    ymin = float(ymin)/1000
    ymax = float(ymax)/1000
    
    return [xmin,xmax,ymin,ymax]

def extract_roiBox_from_metadata_multisensor(metadata, sensor_type):
    
    xmin, xmax, ymin, ymax = 0,0,0,0
    
    if sensor_type == 'scanner3DTop':
        gwu_meta = metadata['cropping_variable_metadata']
        field_bbox_mm = gwu_meta['field_bbox_mm']
        xmin = field_bbox_mm['x']["min"]/1000
        xmax = field_bbox_mm['x']["max"]/1000
        ymin = field_bbox_mm['y']["min"]/1000
        ymax = field_bbox_mm['y']['max']/1000
        
    if sensor_type == 'stereoTop' or sensor_type == 'flirIrCamera':
        gwu_meta = metadata['gwu_added_metadata']
        xmin = gwu_meta["xmin"]
        xmax = gwu_meta["xmax"]
        ymin = gwu_meta["ymin"]
        ymax = gwu_meta['ymax']

    return [xmin,xmax,ymin,ymax]

def find_input_files(in_dir):
    
    metadata_suffix = '.json'
    metas = [meta for meta in glob(join(in_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return

    txt_suffix = '.txt'
    txts = [txtfile for txtfile in glob(join(in_dir, '*' + txt_suffix))]
    if len(txts) == 0:
        return
    
    img_suffix = '.png'
    imgs = [img for img in glob(join(in_dir, '*' + img_suffix))]
    if len(imgs) == 0:
        return

    return [metas, txts, imgs]
    
def parse_position_from_txt(txt_path):
    
    if not os.path.isfile(txt_path):
        return
    
    plantList_img = []
    text_file = open(txt_path, 'r')
    while True:
        line = text_file.readline()
        if line == '':
            break
        fields = line.split(',')
        plantIndex = int(fields[0])
        xmin = int(fields[1])
        ymin = int(fields[2])
        xmax = int(fields[3])
        ymax = int(fields[4])
        plantList_img.append([plantIndex, xmin, ymin, xmax, ymax])
        
    text_file.close()
    
    return plantList_img

def translate_plant_position(plant_position_img, roi_field_boundary, height, width):
        
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin, ymin, xmax, ymax = plant_position_img
    xmin_f = roi_field_boundary[1]-ymax*x_dist
    xmax_f = roi_field_boundary[1]-ymin*x_dist
    ymin_f = roi_field_boundary[3]-xmax*y_dist
    ymax_f = roi_field_boundary[3]-xmin*y_dist
    
    return [plantIndex, xmin_f, xmax_f, ymin_f, ymax_f]

def parse_plant_position_from_input_files(in_files):
    
    json_path = in_files[0][0]
    txt_path = in_files[1][0]
    img_path = in_files[2][0]
    
    metadata = load_json(json_path)
    roi_field_boundary = extract_roiBox_from_metadata(metadata)
    
    img = cv2.imread(img_path)
    height,width = img.shape[:2]
    
    plantList_img = parse_position_from_txt(txt_path)
    if plantList_img == None:
        return
    
    plantList_field = []
    for plant_position in plantList_img:
        plant_field_position = translate_plant_position(plant_position, roi_field_boundary, height, width)
        plantList_field.append(plant_field_position)
    
    return plantList_field

def parse_plantInfo_from_indir(in_dir):
    
    in_files = find_input_files(in_dir)
    
    plantList = parse_plant_position_from_input_files(in_files)
    
    plot_id = os.path.basename(in_dir)
    rel = [plot_id, plantList]
    
    return rel

def crop_rgb_imageToSinglePlant_from_cropDir(plot_path, out_path, plantList, convt):
    
    
    
    return

def find_files_in_crop_dir(plot_dir):
    
    fileList = []
    
    metadata_suffix = '.json'
    metas = [os.path.basename(meta) for meta in glob(join(plot_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return fileList
    
    for meta in metas:
        img_file = meta[:-4]+'png'
        fileList.append([meta, img_file])
    
    return fileList

def find_files_in_crop_dir_laser(plot_dir):
    
    fileList = []
    
    metadata_suffix = '.json'
    metas = [os.path.basename(meta) for meta in glob(join(plot_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return fileList
    
    for meta in metas:
        img_file = meta[:-13]+'g.png'
        fileList.append([meta, img_file])
    
    return fileList

def find_files_in_crop_dir_multisensor(plot_dir, sensor_type):
    
    fileList = []
    
    metadata_suffix = '.json'
    metas = [os.path.basename(meta) for meta in glob(join(plot_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return fileList
    
    for meta in metas:
        if sensor_type == 'scanner3DTop':
            img_file = meta[:-13]+'g.png'
        if sensor_type == 'stereoTop':
            img_file = meta[:-4]+'png'
        if sensor_type == 'flirIrCamera':
            img_file = meta[:-4]+'png'
        if sensor_type == '':
            continue
        fileList.append([meta, img_file])
    
    return fileList

def check_if_target_area_within_image_area(roi_field_boundary, plantList):
    
    for plant_box_field in plantList:
        plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = plant_box_field
        if roi_field_boundary[0]<xmin_f and roi_field_boundary[1]>xmax_f:
            if roi_field_boundary[2]<ymin_f and roi_field_boundary[3]>ymax_f:
                return True
    
    return False

def check_if_target_area_within_image_roi(roi_field_boundary, roi):
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    if roi_field_boundary[0]<xmin_f and roi_field_boundary[1]>xmax_f:
        if roi_field_boundary[2]<ymin_f and roi_field_boundary[3]>ymax_f:
            return True
    return False
    

def crop_rgb_imageToSinglePlant(file_dir, file_pair, out_path, plantList, convt):
    
    json_file, img_file = file_pair
    
    json_full_path = os.path.join(file_dir, json_file)
    img_full_path = os.path.join(file_dir, img_file)
    
    if not os.path.isfile(json_full_path) or not os.path.isfile(img_full_path):
        return
    
    # check if target area is in the img bounds
    metadata = load_json(json_full_path)
    roi_field_boundary = extract_roiBox_from_metadata(metadata)
    if not check_if_target_area_within_image_area(roi_field_boundary, plantList):
        return
    
    # crop plant to target folder
    crop_plant_to_target_folder_RGB(roi_field_boundary, plantList, img_full_path, out_path)
    
    return

def crop_plant_to_target_folder_RGB(roi_field_boundary, plantList, img_full_path, out_path):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    for plant_box_field in plantList:
        plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = plant_box_field
        xmin = round((roi_field_boundary[3]-ymax_f)/x_dist)
        ymin = round((roi_field_boundary[1]-xmax_f)/y_dist)
        xmax = round((roi_field_boundary[3]-ymin_f)/x_dist)
        ymax = round((roi_field_boundary[1]-xmin_f)/y_dist)
        
        if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
            # save roi to file
            out_dir = os.path.join(out_path, str(plantIndex))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            roi_img = img[ymin:ymax, xmin:xmax]
            out_file_name = '{}_{}.png'.format(img_basename, plantIndex)
            out_file_path = os.path.join(out_dir, out_file_name)
            cv2.imwrite(out_file_path, roi_img)
        
    return

def crop_plant_to_target_folder_from_roi_RGB(roi_field_boundary, roi, img_full_path, out_path):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    xmin = int(round((roi_field_boundary[3]-ymax_f)/x_dist))
    ymin = int(round((roi_field_boundary[1]-xmax_f)/y_dist))
    xmax = int(round((roi_field_boundary[3]-ymin_f)/x_dist))
    ymax = int(round((roi_field_boundary[1]-xmin_f)/y_dist))
    
    if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
        # save roi to file
        roi_img = img[ymin:ymax, xmin:xmax]
        out_file_name = 'R_{}_{}.png'.format(plantIndex, img_basename)
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, roi_img)

    return

def crop_plant_to_target_folder_from_roi_thermal(roi_field_boundary, roi, img_full_path, out_path):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    xmin = int(round((roi_field_boundary[3]-ymax_f)/x_dist)) + THERMAL_SHIFT_X
    ymin = int(round((roi_field_boundary[1]-xmax_f)/y_dist)) + THERMAL_SHIFT_Y
    xmax = int(round((roi_field_boundary[3]-ymin_f)/x_dist)) + THERMAL_SHIFT_X
    ymax = int(round((roi_field_boundary[1]-xmin_f)/y_dist)) + THERMAL_SHIFT_Y
    
    if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
        # save roi to file
        roi_img = img[ymin:ymax, xmin:xmax]
        out_file_name = 'T_{}_{}.png'.format(plantIndex, img_basename)
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, roi_img)

    return

def crop_plant_to_target_folder_from_roi_stereoTop(roi_field_boundary, roi, img_full_path, out_dir, str_plot_id):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    xmin = int(round((roi_field_boundary[3]-ymax_f)/x_dist))
    ymin = int(round((roi_field_boundary[1]-xmax_f)/y_dist))
    xmax = int(round((roi_field_boundary[3]-ymin_f)/x_dist))
    ymax = int(round((roi_field_boundary[1]-xmin_f)/y_dist))
    
    if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
            # which date
        str_date = img_basename[:10]
        out_path = os.path.join(out_dir, str_date, str_plot_id)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        # save roi to file
        roi_img = img[ymin:ymax, xmin:xmax]
        out_file_name = 'R_{}_{}.png'.format(plantIndex, img_basename)
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, roi_img)
        
    return


def crop_plant_to_target_folder_from_roi_flirlrCamera(roi_field_boundary, roi, img_full_path, out_dir, str_plot_id):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    xmin = int(round((roi_field_boundary[3]-ymax_f)/x_dist)) + THERMAL_SHIFT_X
    ymin = int(round((roi_field_boundary[1]-xmax_f)/y_dist)) + THERMAL_SHIFT_Y
    xmax = int(round((roi_field_boundary[3]-ymin_f)/x_dist)) + THERMAL_SHIFT_X
    ymax = int(round((roi_field_boundary[1]-xmin_f)/y_dist)) + THERMAL_SHIFT_Y
    
    if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
        # which date
        str_date = img_basename[:10]
        out_path = os.path.join(out_dir, str_date, str_plot_id)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        
        # save roi to file
        roi_img = img[ymin:ymax, xmin:xmax]
        out_file_name = 'T_{}_{}.png'.format(plantIndex, img_basename)
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, roi_img)

    return

def render_to_vertical_view(g_file, j_file, np_file, g_vertical_view_path):
    
    plyData = np.load(np_file)
    #np_xy = np.nan_to_num(plyData)
    
    sensor_type = 'scanner3DTop'
    metadata = load_json(j_file)
    roi_field_boundary = extract_roiBox_from_metadata_multisensor(metadata, sensor_type)
    
    gImg = cv2.imread(g_file, -1)
    height,width = gImg.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/width
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/height

    np_remap = np.zeros((height, width, 2))
    
    field_x0 = roi_field_boundary[0]
    field_y0 = roi_field_boundary[2]
    
    for i in range(0, height):
        for j in range(0, width):
            c_point = plyData[i, j]
            if np.isnan(c_point).any():
                continue
            n_idx_x = int(round((c_point[0]/1000 - field_x0)/x_dist))
            n_idx_y = int(round((c_point[1]/1000 - field_y0)/y_dist))
            if n_idx_x < 0 or n_idx_x > width-1 or n_idx_y < 0 or n_idx_y > height-1:
                continue
            
            if c_point[2] > np_remap[n_idx_y, n_idx_x, 0]:
                np_remap[n_idx_y, n_idx_x, 0] = c_point[2]
                np_remap[n_idx_y, n_idx_x, 1] = gImg[i, j]
    
    save_img = np_remap[:,:,1]
    save_img = save_img.astype('uint8')
    mask = np.zeros((height, width))
    mask[save_img == 0] = 1
    
    median = cv2.medianBlur(save_img,3)
    median = median * mask
    out_img = median + save_img
    
    cv2.imwrite(g_vertical_view_path, out_img)
    
    return

def crop_plant_to_target_folder_from_roi_LASER(roi_field_boundary, roi, img_full_path, out_dir, str_plot_id):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    im1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    height,width = im1.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = roi
    xmin = int(round((roi_field_boundary[3]-ymax_f)/y_dist)) + THERMAL_SHIFT_Y
    ymin = int(round((roi_field_boundary[1]-xmax_f)/x_dist)) + THERMAL_SHIFT_X
    xmax = int(round((roi_field_boundary[3]-ymin_f)/y_dist)) + THERMAL_SHIFT_Y
    ymax = int(round((roi_field_boundary[1]-xmin_f)/x_dist)) + THERMAL_SHIFT_X
    
    if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
        # which date
        str_date = img_basename[:10]
        out_path = os.path.join(out_dir, str_date, str_plot_id)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        
        # save roi to file
        roi_img = im1[ymin:ymax, xmin:xmax]
        out_file_name = 'L_{}_{}.png'.format(plantIndex, img_basename)
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, roi_img)
    
    return

def crop_plant_to_target_folder_thermal(roi_field_boundary, plantList, img_full_path, out_path):
    
    img_basename = os.path.basename(img_full_path)[:-4]
    
    img = cv2.imread(img_full_path)
    height,width = img.shape[:2]
    
    x_dist = (roi_field_boundary[1]-roi_field_boundary[0])/height
    y_dist = (roi_field_boundary[3]-roi_field_boundary[2])/width
    
    for plant_box_field in plantList:
        plantIndex, xmin_f, xmax_f, ymin_f, ymax_f = plant_box_field
        xmin = round((roi_field_boundary[3]-ymax_f)/x_dist) + THERMAL_SHIFT_X
        ymin = round((roi_field_boundary[1]-xmax_f)/y_dist) + THERMAL_SHIFT_Y
        xmax = round((roi_field_boundary[3]-ymin_f)/x_dist) + THERMAL_SHIFT_X
        ymax = round((roi_field_boundary[1]-xmin_f)/y_dist) + THERMAL_SHIFT_Y
        
        if xmin > 0 and xmax < width and ymin > 0 and ymax < height:
            # save roi to file
            out_dir = os.path.join(out_path, str(plantIndex))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            roi_img = img[ymin:ymax, xmin:xmax]
            out_file_name = '{}_{}.png'.format(img_basename, plantIndex)
            out_file_path = os.path.join(out_dir, out_file_name)
            cv2.imwrite(out_file_path, roi_img)
        
    return

def crop_thermal_imageToSinglePlant(plot_path, out_path, plantList, convt):
    
    fileList = find_files_in_crop_dir(plot_path)
    
    for file_pair in fileList:
        json_file, img_file = file_pair
    
        json_full_path = os.path.join(plot_path, json_file)
        img_full_path = os.path.join(plot_path, img_file)
    
        if not os.path.isfile(json_full_path) or not os.path.isfile(img_full_path):
            continue
    
        # check if target area is in the img bounds
        metadata = load_json(json_full_path)
        roi_field_boundary = extract_roiBox_from_metadata(metadata)
        if not check_if_target_area_within_image_area(roi_field_boundary, plantList):
            continue
        
        # crop plant to target folder
        crop_plant_to_target_folder_thermal(roi_field_boundary, plantList, img_full_path, out_path)
    
    return

def full_season_singlePlantCrop_rgb_from_cropDir(in_dir, plot_dir, out_dir, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    plantInfo = parse_plantInfo_from_indir(in_dir)
    
    plot_id = plantInfo[0]
    plantList = plantInfo[1]  #plant_id, plant_field_roi
    
    out_path = os.path.join(out_dir, plot_id)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    # loop all files
    fileList = find_files_in_crop_dir(plot_dir)
    
    for file_pair in fileList:
        crop_rgb_imageToSinglePlant(plot_dir, file_pair, out_path, plantList, convt)
    
    return

def full_season_singlePlantCrop_thermal(in_dir, plot_dir, out_dir, start_date, end_date, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    plantInfo = parse_plantInfo_from_indir(in_dir)
    
    plot_id = plantInfo[0]
    plantList = plantInfo[1]  #plant_id, plant_field_roi
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        plot_path = os.path.join(plot_dir, str_date, plot_id)
        if not os.path.isdir(plot_path):
            continue
        
        out_path = os.path.join(out_dir, plot_id)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        crop_thermal_imageToSinglePlant(plot_path, out_path, plantList, convt)
        #full_day_multi_process(raw_path, out_path, plot_dir, convt)
    
    return

def generate_roi_list_all_plots(crop_size, convt):
    
    rows, cols = (convt.max_range, convt.max_col) 
    roi_list = [[[] for i in range(cols)] for j in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            target_bounds = convt.np_bounds[i][j]
            
            x_length = int((target_bounds[1]-target_bounds[0])//crop_size)
            y_length = int((target_bounds[3]-target_bounds[2])//crop_size)
            
            x_start = target_bounds[0]
            y_start = target_bounds[2]
            
            plot_list = []
            ind = 0
            for x in range(x_length):
                for y in range(y_length):
                    x0 = x_start + crop_size*x
                    x1 = x_start + crop_size*(x+1) 
                    y0 = y_start + crop_size*y
                    y1 = y_start + crop_size*(y+1)
                   
                    plot_list.append([ind,x0,x1,y0,y1])
                    ind += 1
                   
            roi_list[i][j] = plot_list

    return roi_list

def generate_roi_list_all_plots_center_plant(crop_size, convt):
    
    rows, cols = (convt.max_range, convt.max_col) 
    roi_list = [[[] for i in range(cols)] for j in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            target_bounds = convt.np_bounds[i][j]
            
            x_length = int((target_bounds[1]-target_bounds[0])//crop_size)
            y_length = int((target_bounds[3]-target_bounds[2])//crop_size)
            
            x_start = target_bounds[0]
            y_start = target_bounds[2]
            
            # center y boundary
            plot_y_dist = (target_bounds[3]-target_bounds[2])
            y_step = plot_y_dist/4
            y1_center = target_bounds[2]+y_step
            y2_center = target_bounds[2]+3*y_step

            plot_list = []
                        
            y0 = y1_center-crop_size/2
            y1 = y1_center+crop_size/2
            ind = 0
            for x in range(x_length):
                x0 = x_start + crop_size*x
                x1 = x_start + crop_size*(x+1)
                plot_list.append([ind,x0,x1,y0,y1])
                ind += 1
                
                x0 = x_start + crop_size*x + crop_size/2
                x1 = x_start + crop_size*(x+1) + crop_size/2
                plot_list.append([ind,x0,x1,y0,y1])
                ind += 1
            
            y0 = y2_center-crop_size/2
            y1 = y2_center+crop_size/2
            for x in range(x_length):
                x0 = x_start + crop_size*x
                x1 = x_start + crop_size*(x+1)
                plot_list.append([ind,x0,x1,y0,y1])
                ind += 1
                
                x0 = x_start + crop_size*x + crop_size/2
                x1 = x_start + crop_size*(x+1) + crop_size/2
                plot_list.append([ind,x0,x1,y0,y1])
                ind += 1

                   
            roi_list[i][j] = plot_list

    return roi_list



def rgb_test(convt):
    
    in_dir = '/media/zli/Elements/test/31_13_493_shortcut_0/31-13-493'
    plot_dir = '/media/zli/Elements/test/31_13_493_shortcut_0/31_13_493_shortcut'
    out_dir = '/media/zli/Elements/test/31_13_493_shortcut_0/plant_dir'
    
    full_season_singlePlantCrop_rgb_from_cropDir(in_dir, plot_dir, out_dir, convt)
    
    return


def thermal_test(convt):
    
    start_date = '2019-06-02'
    end_date = '2019-06-02'
    
    in_dir = '/media/zli/Elements/test/31_13_493_shortcut_0/31-13-493'
    plot_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot_png'
    out_dir = '/media/zli/Elements/test/31_13_493_shortcut_0/thermal_plant_dir'
    
    full_season_singlePlantCrop_thermal(in_dir, plot_dir, out_dir, start_date, end_date, convt)
    
    return

def crop_by_size_single_sensor(sub_in_dir, sub_out_dir, rois, sensor='R'):
    
    fileList = find_files_in_crop_dir(sub_in_dir)
    
    # loop roi_list
    for roi in rois:
        # loop rgb files
        for file_pair in fileList:
            json_file, img_file = file_pair
            
            json_full_path = os.path.join(sub_in_dir, json_file)
            img_full_path = os.path.join(sub_in_dir, img_file)
            
            if not os.path.isfile(json_full_path) or not os.path.isfile(img_full_path):
                continue
            
            # check if target area is in the img bounds
            metadata = load_json(json_full_path)
            roi_field_boundary = extract_roiBox_from_metadata(metadata)
            if not check_if_target_area_within_image_roi(roi_field_boundary, roi):
                continue
            
            # crop plant to target folder
            if sensor == 'R':
                crop_plant_to_target_folder_from_roi_RGB(roi_field_boundary, roi, img_full_path, sub_out_dir)
            if sensor == 'T':
                crop_plant_to_target_folder_from_roi_thermal(roi_field_boundary, roi, img_full_path, sub_out_dir)
            break
    
    return

def crop_by_size_multisensor(in_dir, out_dir, rois, str_plot_id, sensor_type):
    
    fileList = find_files_in_crop_dir_multisensor(in_dir, sensor_type)
    
    out_path = os.path.join(out_dir, sensor_type)
    
    ind = 0
    
    # loop roi_list
    for roi in rois:
        #loop img files
        for file_pair in fileList:
            json_file, img_file = file_pair
            json_full_path = os.path.join(in_dir, json_file)
            img_full_path = os.path.join(in_dir, img_file)
            
            if not os.path.isfile(json_full_path) or not os.path.isfile(img_full_path):
                continue
            
            # check if target area is in the img bounds
            metadata = load_json(json_full_path)
            roi_field_boundary = extract_roiBox_from_metadata_multisensor(metadata, sensor_type)
            if not check_if_target_area_within_image_roi(roi_field_boundary, roi):
                continue
            
            ind += 1
            # crop plant to target folder
            if sensor_type == 'stereoTop':
                crop_plant_to_target_folder_from_roi_stereoTop(roi_field_boundary, roi, img_full_path, out_path, str_plot_id)
                break
            if sensor_type == 'flirIrCamera':
                crop_plant_to_target_folder_from_roi_flirlrCamera(roi_field_boundary, roi, img_full_path, out_path, str_plot_id)
                break
            if sensor_type == 'scanner3DTop':
                crop_plant_to_target_folder_from_roi_LASER(roi_field_boundary, roi, img_full_path, out_path, str_plot_id)
                
    print(ind)
    
    return


def crop_by_size_laser(in_dir, out_dir, rois, str_plot_id, sensor_type, temp_dir):
    '''
    fileList = find_files_in_crop_dir_laser(in_dir)
    
    # loop roi_list
    for roi in rois:
        #loop img files
        for file_pair in fileList:
            json_file, img_file = file_pair
            json_full_path = os.path.join(in_dir, json_file)
            img_full_path = os.path.join(in_dir, img_file)
            
            if not os.path.isfile(json_full_path) or not os.path.isfile(img_full_path):
                continue
            
            # check if target area is in the img bounds
            metadata = load_json(json_full_path)
            roi_field_boundary = extract_roiBox_from_metadata_laser(metadata)
            if not check_if_target_area_within_image_roi(roi_field_boundary, roi):
                continue
            
            # crop plant to target folder
            crop_plant_to_target_folder_from_roi_LASER(roi_field_boundary, roi, img_full_path, out_dir, str_plot_id)
    '''
    fileList = find_files_in_crop_dir_multisensor(in_dir, sensor_type)
    
    out_path = os.path.join(out_dir, sensor_type)
    
    ind = 0
    
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    
    # loop roi_list
    for roi in rois:
        #loop img files
        for file_pair in fileList:
            json_file, img_file = file_pair
            json_full_path = os.path.join(in_dir, json_file)
            g_img_full_path = os.path.join(in_dir, img_file)
            g_vertical_view_path = os.path.join(temp_dir, img_file)
            p_img_full_path = g_img_full_path[:-5]+'p.png'
            zip_np_file = g_img_full_path[:-5] + 'xyz.npy.gz'
            
            
            if not os.path.isfile(json_full_path) or not os.path.isfile(g_img_full_path) or not os.path.isfile(p_img_full_path) or not os.path.isfile(zip_np_file):
                continue
            
            # check if target area is in the img bounds
            metadata = load_json(json_full_path)
            roi_field_boundary = extract_roiBox_from_metadata_multisensor(metadata, sensor_type)
            if not check_if_target_area_within_image_roi(roi_field_boundary, roi):
                continue
            
            ind += 1
            # copy files to temp_dir, unzip gz file
            np_file = unzip_np_files_to_temp_dir(zip_np_file, temp_dir)
            
            # create vertical view to temp_dir
            render_to_vertical_view(g_img_full_path, json_full_path, np_file, g_vertical_view_path)
            
            # crop to plant and save file
            crop_plant_to_target_folder_from_roi_LASER(roi_field_boundary, roi, g_vertical_view_path, out_path, str_plot_id)
            
            # delete temp files
            try:
                if os.path.isfile(np_file):
                    os.remove(np_file)
                if os.path.isfile(g_vertical_view_path):
                    os.remove(g_vertical_view_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (np_file, e))
    print(ind)
    return


def unzip_np_files_to_temp_dir(zip_np_file, temp_dir):
    
    dst_npy_file = os.path.join(temp_dir, os.path.basename(zip_np_file)[:-3])
    with gzip.open(zip_np_file, 'rb') as f_in:
        with open(dst_npy_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return dst_npy_file

def delete_temp_files(temp_dir):

    folder = temp_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    return

def crop_laser_to_roi(roi_list, in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    # loop each cultivar
    list_cultivar_dirs = os.listdir(in_dir)
    for d in list_cultivar_dirs:
        cultivar_dir = os.path.join(in_dir, d)
        if not os.path.isdir(cultivar_dir):
            continue
        # loop plot
        list_plot_dirs = os.listdir(cultivar_dir)
        for plot_d in list_plot_dirs:
            plot_dir = os.path.join(cultivar_dir, plot_d)
            if not os.path.isdir(plot_dir):
                continue
            
            # get plotID
            plot_id = [int(s) for s in plot_d.split('_') if s.isdigit()]

            str_plot_id = '{0:02d}-{1:02d}-{2:04d}'.format(plot_id[0], plot_id[1], plot_id[2])
            
            # get roi list
            rois = roi_list[plot_id[0]-1][plot_id[1]-1]
            
            plot_3d_dir = os.path.join(plot_dir, 'scanner3DTop')
            if not os.path.isdir(plot_3d_dir):
                continue
            
            # crop file to output
            crop_by_size_laser(plot_3d_dir, out_dir, rois, str_plot_id)
    
    return

def crop_to_roi_multisensor(roi_list, in_dir, out_dir, sensor_type, temp_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    # loop each cultivar
    list_cultivar_dirs = os.listdir(in_dir)
    for d in list_cultivar_dirs:
        cultivar_dir = os.path.join(in_dir, d)
        if not os.path.isdir(cultivar_dir):
            continue
        
        print(d)
        # loop plot
        list_plot_dirs = os.listdir(cultivar_dir)
        for plot_d in list_plot_dirs:
            plot_dir = os.path.join(cultivar_dir, plot_d)
            if not os.path.isdir(plot_dir):
                continue
            
            # get plotID
            plot_id = [int(s) for s in plot_d.split('_') if s.isdigit()]
            str_plot_id = '{0:02d}-{1:02d}-{2:04d}'.format(plot_id[0], plot_id[1], plot_id[2])
            
            print(plot_d)
            # get roi list
            rois = roi_list[plot_id[0]-1][plot_id[1]-1]
            plot_sensor_dir = os.path.join(plot_dir, sensor_type)
            if not os.path.isdir(plot_sensor_dir):
                continue
            
            # crop file to output
            #crop_by_size_multisensor(plot_sensor_dir, out_dir, rois, str_plot_id, sensor_type)
            crop_by_size_laser(plot_sensor_dir, out_dir, rois, str_plot_id, sensor_type, temp_dir)
    
    return

def crop_to_roi_multisensor_multi_process(roi_list, in_dir, out_dir, sensor_type, temp_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    list_dirs = [os.path.join(in_dir,o) for o in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,o))]
    numDirs = len(list_dirs)
    
    print ("Starting ply to npy conversion...")
    pool = multiprocessing.Pool()
    NUM_THREADS = min(20,numDirs)
    print('numDirs:{}   NUM_THREADS:{}'.format(numDirs, NUM_THREADS))
    for cpu in range(NUM_THREADS):
        pool.apply_async(sub_process, [list_dirs[cpu::NUM_THREADS], roi_list, out_dir, sensor_type, temp_dir])
    pool.close()
    pool.join()
    print ("Completed ply to npy conversion...")
    
    return

def sub_process(cultivar_dir, roi_list, out_dir, sensor_type, temp_dir):
    
    
    for d in cultivar_dir:
        #Generate jpgs and geoTIFs from .bin
        try:
            print(d)
            if not os.path.isdir(d):
                return
            # loop plot
            list_plot_dirs = os.listdir(d)
            for plot_d in list_plot_dirs:
                plot_dir = os.path.join(d, plot_d)
                if not os.path.isdir(plot_dir):
                    continue
                
                # get plotID
                plot_id = [int(s) for s in plot_d.split('_') if s.isdigit()]
                str_plot_id = '{0:02d}-{1:02d}-{2:04d}'.format(plot_id[0], plot_id[1], plot_id[2])
                
                # get roi list
                rois = roi_list[plot_id[0]-1][plot_id[1]-1]
                plot_sensor_dir = os.path.join(plot_dir, sensor_type)
                if not os.path.isdir(plot_sensor_dir):
                    continue
                
                crop_by_size_laser(plot_sensor_dir, out_dir, rois, str_plot_id, sensor_type, temp_dir)
            #bin_to_geotiff.stereo_test(s, s)
        except Exception as ex:
            print("\tFailed to process folder %s: %s" % (d, str(ex)))

    return

def crop_from_both_sensors(roi_list, in_rgb_dir, in_thermal_dir, start_date, end_date, out_rgb_dir, out_thermal_dir, convt):
    
    if not os.path.isdir(out_rgb_dir):
        os.makedirs(out_rgb_dir)
        
    if not os.path.isdir(out_thermal_dir):
        os.makedirs(out_thermal_dir)
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    # loop each day
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        rgb_day_dir = os.path.join(in_rgb_dir, str_date)
        thermal_day_dir = os.path.join(in_thermal_dir, str_date)
        
        out_day_rgb = os.path.join(out_rgb_dir, str_date)
        out_day_thermal = os.path.join(out_thermal_dir, str_date)
        
        if not os.path.isdir(rgb_day_dir) or not os.path.isdir(thermal_day_dir):
            continue
        
        if not os.path.isdir(out_day_rgb):
            os.mkdir(out_day_rgb)
        if not os.path.isdir(out_day_thermal):
            os.mkdir(out_day_thermal)
        
        list_rgb_sub_dirs = os.listdir(rgb_day_dir)
        for d in list_rgb_sub_dirs:
            if len(d) == 9:
                continue
            sub_rgb_dir = os.path.join(rgb_day_dir, d)
            if not os.path.isdir(sub_rgb_dir):
                continue

            # get roi_list
            plot_id = [int(s) for s in d.split('_') if s.isdigit()]
            rois = roi_list[plot_id[0]-1][plot_id[1]-1]
            
            sub_rgb_out = os.path.join(out_day_rgb, d)
            if not os.path.isdir(sub_rgb_out):
                os.mkdir(sub_rgb_out)
            # crop file
            crop_by_size_single_sensor(sub_rgb_dir, sub_rgb_out, rois, sensor='R')
            
        list_thermal_sub_dirs = os.listdir(thermal_day_dir)
        for d in list_thermal_sub_dirs:
            if len(d) == 9:
                continue
            sub_thermal_dir = os.path.join(thermal_day_dir, d)
            if not os.path.isdir(sub_thermal_dir):
                continue
            
            # get roi_list
            plot_id = [int(s) for s in d.split('_') if s.isdigit()]
            rois = roi_list[plot_id[0]-1][plot_id[1]-1]
            
            sub_thermal_out = os.path.join(out_day_thermal, d)
            if not os.path.isdir(sub_thermal_out):
                os.mkdir(sub_thermal_out)
                
            crop_by_size_single_sensor(sub_thermal_dir, sub_thermal_out, rois, sensor='T')
    
    return

def stereo_resize_reorganize(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    start_date = '2019-05-20'  # S9 start date
    end_date = '2019-06-20'   # S9 end date
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    # loop each day
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        rgb_day_dir = os.path.join(in_dir, str_date)
        if not os.path.isdir(rgb_day_dir):
            continue
        
        list_rgb_sub_dirs = os.listdir(rgb_day_dir)
        for d in list_rgb_sub_dirs:
            folder_split = d.split('-')
            if not len(folder_split[2]) == 4:
                continue
            sub_rgb_dir = os.path.join(rgb_day_dir, d)
            if not os.path.isdir(sub_rgb_dir):
                continue
            
            file_list = glob(os.path.join(sub_rgb_dir, '*{}'.format('png')))
            for file_path in file_list:
                base_name = os.path.basename(file_path)
                file_name_split = base_name.split('_')
                crop_id = file_name_split[1]
                
                out_sub_dir = os.path.join(out_dir, '{}-{}'.format(d, crop_id))
                if not os.path.isdir(out_sub_dir):
                    os.mkdir(out_sub_dir)
                    
                out_file_path = os.path.join(out_sub_dir, '{}.png'.format(str_date))
                img = cv2.imread(file_path, 1)
                resized_img = cv2.resize(img, (400,400), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(out_file_path, resized_img)
            
    return

def reorganize_main():
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermal_fusion_data_0.5'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/thermal_crop_resized_0.5'
    
    #in_dir = '/data/shared/sensorFusion/sensorFusion_preprocess_vertical_view/scanner3DTop'
    #out_dir = '/data/shared/sensorFusion/sensorFusion_resize/scanner3DTop_verticalView'
    stereo_resize_reorganize(in_dir, out_dir)
    remove_folder(out_dir)
    return

import shutil, random

def init_joint_data(rgb_dir, thermal_dir, out_rgb, out_thermal):
    
    if not os.path.isdir(out_rgb):
        os.makedirs(out_rgb)
    if not os.path.isdir(out_thermal):
        os.makedirs(out_thermal)
    
    # loop rgb_dir
    list_rgb_dir = os.listdir(rgb_dir)
    
    for d in list_rgb_dir:
        sub_rgb_dir = os.path.join(rgb_dir, d)
        sub_thermal_dir = os.path.join(thermal_dir, d)
        # if both dir exist
        if os.path.isdir(sub_rgb_dir) and os.path.isdir(sub_thermal_dir):
            # copy rgb dir
            dst_rgb_dir = os.path.join(out_rgb, d)
            shutil.copytree(sub_rgb_dir, dst_rgb_dir)
            
            # copy thermal dir
            dst_thermal_dir = os.path.join(out_thermal, d)
            shutil.copytree(sub_thermal_dir, dst_thermal_dir)
    
    return

def init_joint_data_hard(rgb_dir, laser_dir, out_rgb, out_laser):
    
    if not os.path.isdir(out_rgb):
        os.makedirs(out_rgb)
    if not os.path.isdir(out_laser):
        os.makedirs(out_laser)
    
    # loop rgb_dir
    list_rgb_dir = os.listdir(rgb_dir)
    
    for d in list_rgb_dir:
        sub_rgb_dir = os.path.join(rgb_dir, d)
        sub_laser_dir = os.path.join(laser_dir, d)
        # if both dir exist
        if os.path.isdir(sub_rgb_dir) and os.path.isdir(sub_laser_dir):
            # loop sub dir, check if same day data exist
            list_sub_rgb_files = os.walk(sub_rgb_dir)
            for root, dirs, files in list_sub_rgb_files:
                for f in files:
                    base_file = os.path.join(sub_rgb_dir, f)
                    if os.path.exists(base_file):
                        pair_file = os.path.join(sub_laser_dir, f)
                        if os.path.exists(pair_file):
                            # check folder exist
                            dst_rgb_dir = os.path.join(out_rgb, d)
                            dst_laser_dir = os.path.join(out_laser, d)
                            if not os.path.isdir(dst_rgb_dir):
                                os.mkdir(dst_rgb_dir)
                            if not os.path.isdir(dst_laser_dir):
                                os.mkdir(dst_laser_dir)
                            
                            # copy file
                            dst_rgb_file = os.path.join(dst_rgb_dir, f)
                            dst_laser_file = os.path.join(dst_laser_dir, f)
                            
                            shutil.copy(base_file, dst_rgb_file)
                            shutil.copy(pair_file, dst_laser_file)
            
    return

def remove_folder(in_dir):
    
    list_dirs = os.listdir(in_dir)
    for d in list_dirs:
        plot_dir = os.path.join(in_dir, d)
        if not os.path.isdir(plot_dir):
            continue
        
        file_list = glob(os.path.join(plot_dir, '*{}'.format('png')))
        if len(file_list) < 2:
            shutil.rmtree(plot_dir)
    
    return

def randomly_move_folder(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    src_rgb_dir = os.path.join(in_dir, 'rgb')
    src_thermal_dir = os.path.join(in_dir, 'thermal')
    
    dst_rgb_dir = os.path.join(out_dir, 'rgb')
    dst_thermal_dir = os.path.join(out_dir, 'thermal')
    if not os.path.isdir(dst_rgb_dir):
        os.makedirs(dst_rgb_dir)
    if not os.path.isdir(dst_thermal_dir):
        os.makedirs(dst_thermal_dir)
    
    list_dirs = os.listdir(src_rgb_dir)
    
    sel_dirs = random.choices(list_dirs, k=70)
    
    for d in sel_dirs:
        src_rgb = os.path.join(src_rgb_dir, d)
        if not os.path.isdir(src_rgb):
            continue
        src_thermal = os.path.join(src_thermal_dir, d)
        if not os.path.isdir(src_thermal):
            continue
        
        dst_rgb = os.path.join(dst_rgb_dir, d)
        shutil.move(src_rgb, dst_rgb)
        dst_thermal = os.path.join(dst_thermal_dir, d)
        shutil.move(src_thermal, dst_thermal)
        
    return

def main_local_workstation():

    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2019-06-18') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    
    roi_list = generate_roi_list_all_plots_center_plant(0.5, convt)
    
    start_date = '2019-06-07'  # S9 start date
    end_date = '2019-06-07'   # S9 end date
    in_rgb_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/rgb_crop_new'
    in_thermal_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot_png_new'
    out_rgb_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/rgb_fusion_data_0.5_new'
    out_thermal_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermal_fusion_data_0.5_new'
    
    crop_from_both_sensors(roi_list, in_rgb_dir, in_thermal_dir, start_date, end_date, out_rgb_dir, out_thermal_dir, convt)
    
    return

def test():
    
    rgb_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/rgb_crop_resized_double'
    thermal_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/thermal_crop_resized_double'
    out_rgb = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_double/rgb'
    out_thermal = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_double/thermal'
    
    init_joint_data(rgb_dir, thermal_dir, out_rgb, out_thermal)
    
    return

def laser_main():
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/laser_crop/2019_06__3-6'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/laser_fusion_data'
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2019-06-18') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    
    roi_list = generate_roi_list_all_plots_center_plant(0.6, convt)
    
    crop_laser_to_roi(roi_list, in_dir, out_dir)
    
    return

def split_val_test_main():
    
    in_dir = '/data/shared/sensorFusion/training_data/joint_rgb_laser_verticalView/train/'
    out_dir = '/data/shared/sensorFusion/training_data/joint_rgb_laser_verticalView/test/'
    randomly_move_folder(in_dir, out_dir)
    out_dir = '/data/shared/sensorFusion/training_data/joint_rgb_laser_verticalView/val'
    randomly_move_folder(in_dir, out_dir)
    
    return

def gwu_crop_to_roi(in_dir, out_dir, sensor_type, bety_query_date, temp_dir, field_size=0.6):
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query(bety_query_date)  # s9, '2019-06-18'
    
    if not qFlag:
        return
    
    roi_list = generate_roi_list_all_plots_center_plant(field_size, convt)
    
    crop_to_roi_multisensor_multi_process(roi_list, in_dir, out_dir, sensor_type, temp_dir)
    
    return


def main():
    '''
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/laser_crop/2019-05-27'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/sensorFusion_preprocess'
    temp_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/temp_dir'
    '''
    in_dir = '/data/shared/season_9'
    out_dir = '/data/shared/sensorFusion/sensorFusion_preprocess_vertical_view'
    temp_dir = '/pless_nfs/home/zongyangli/temp_dir'
    sensor_type = 'scanner3DTop'  # stereoTop  scanner3DTop flirIrCamera
    bety_query_date = '2019-06-18'
    field_size = 0.6
    
    gwu_crop_to_roi(in_dir, out_dir, sensor_type, bety_query_date, temp_dir, field_size)
    
    return

def init_joint_main():
    '''
    rgb_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/rgb_crop_resized_double'
    laser_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/thermal_crop_resized_double'
    out_rgb = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_rgb_thermal_hard/train/rgb'
    out_laser = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_rgb_thermal_hard/train/thermal'
    '''
    
    rgb_dir = '/data/shared/sensorFusion/sensorFusion_resize/rgb_crop_resized_double'
    laser_dir = '/data/shared/sensorFusion/sensorFusion_resize/scanner3DTop_verticalView'
    out_rgb = '/data/shared/sensorFusion/training_data/joint_rgb_laser_verticalView/train/rgb'
    out_laser = '/data/shared/sensorFusion/training_data/joint_rgb_laser_verticalView/train/laser'
    
    init_joint_data(rgb_dir, laser_dir, out_rgb, out_laser)
    
    return


if __name__ == '__main__':
    
    main_local_workstation()
    reorganize_main()
    init_joint_main()
    split_val_test_main()
    #test()
    #laser_main()
    
    
    
    
    
    
    
    
