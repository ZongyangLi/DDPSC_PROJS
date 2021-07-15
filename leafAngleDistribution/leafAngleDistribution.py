'''
Created on Nov 17, 2016

@author: Zongyang
'''

import os, sys, terra_common, argparse, shutil, math, colorsys, json
import cv2
import numpy as np
import multiprocessing
from numpy import linspace
from glob import glob
from plyfile import PlyData, PlyElement
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
import matplotlib.pyplot as plt
from lmfit import Model
from datetime import date, timedelta, datetime
#from terrautils import betydb, spatial

#PLOT_RANGE_NUM = 54
#PLOT_COL_NUM = 16

os.environ['BETYDB_KEY'] = 'GZJZWnJpnDBhmKk7k6vb46z6lW6vjxSniRivRl2I'
CPUS = 28

def options():
    
    parser = argparse.ArgumentParser(description='Angle Distribution Extractor in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="all day flag, all for all day process, given parent directory as input, one for one day process")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-v", "--save_dir", help="integrate result in another output directory")
    parser.add_argument("-b", "--bety_dir", help="output bety csv dir")

    args = parser.parse_args()

    return args

def process_all_scanner_data(ply_parent, json_parent, out_parent, save_dir_parent):
    
    list_dirs = os.listdir( ply_parent )
    
    for str_date in list_dirs:
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        save_path = os.path.join(save_dir_parent, str_date)
        if not os.path.isdir(ply_path):
            print(ply_path)
            continue
        if not os.path.isdir(json_path):
            print(json_path)
            continue
    
        print('start processing ' + str_date)
        convt = terra_common.CoordinateConverter()
        try:
            q_flag = convt.bety_query(str_date, True)
            if not q_flag:
                print('Bety query failed')
                continue
            
            full_day_gen_angle_multi_process(str_date, ply_path, json_path, out_path, convt)
            
            full_day_summary(out_path, save_path, convt)
    
        except Exception as ex:
            fail(str(ex))

    
    return

def full_day_gen_angle_multi_process(str_date, ply_path, json_path, out_path, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = [os.path.join(ply_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    json_dirs = [os.path.join(json_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    out_dirs = [os.path.join(out_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    numDirs = len(list_dirs)
    
    print ('Starting ply to npy conversion...')
    pool = multiprocessing.Pool()
    NUM_THREADS = min(CPUS,numDirs)
    for cpu in range(NUM_THREADS):
        pool.apply_async(ply_to_npy, [list_dirs[cpu::NUM_THREADS], json_dirs[cpu::NUM_THREADS], out_dirs[cpu::NUM_THREADS], convt, str_date])
    pool.close()
    pool.join()
    print ("Completed ply to npy conversion...")
    
    return

def ply_to_npy(ply_dirs, json_dirs, out_dirs, convt, str_date):
    for p, j, o in zip(ply_dirs, json_dirs, out_dirs):
        #Generate angles from .ply
        try:
            create_angle_data(str_date, p, j, o, convt)
        except Exception as ex:
            fail("\tFailed to process folder %s: %s" % (p, str(ex)))

# main function
def process_angleData_by_date(ply_parent, json_parent, out_parent, str_year, str_start_month, str_end_month, str_start_date, str_end_date, save_dir_parent):
    
    if not os.path.isdir(save_dir_parent):
        os.makedirs(save_dir_parent)
        
    d1 = date(int(str_year), int(str_start_month), int(str_start_date))
    d2 = date(int(str_year), int(str_end_month), int(str_end_date))
    
    deltaDay = d2 - d1
        
    for i in range(deltaDay.days+1):
        str_date = str(d1+timedelta(days=i))
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        save_path = os.path.join(save_dir_parent, str_date)
        
        if not os.path.isdir(ply_path):
            continue
        if not os.path.isdir(json_path):
            continue
        
        convt = terra_common.CoordinateConverter()
        try:
            q_flag = convt.bety_query(str_date, True)
            if not q_flag:
                print('Bety query for boundaries failed')
                continue
            
            # create one days leaf angle by plot
            full_day_gen_angle_data(str_date, ply_path, json_path, out_path, convt)
    
            # full days angle hist summary
            full_day_summary(out_path, save_path, convt)
            
            # data uploading
            insert_leafAngle_traits_into_betydb(save_path, save_path, str_date, convt)
        except Exception as ex:
            fail(str_date + str(ex))
    
    return

def full_day_gen_angle_data(str_date, ply_path, json_path, out_path, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = os.walk(ply_path)
    
    print('ply: {}, json: {}, out: {}'.format(ply_path, json_path, out_path))
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            p_path = os.path.join(ply_path, d)
            j_path = os.path.join(json_path, d)
            o_path = os.path.join(out_path, d)
            if not os.path.isdir(p_path):
                continue
            
            if not os.path.isdir(j_path):
                continue
            
            create_angle_data(str_date, p_path, j_path, o_path, convt)
            #color_code_depth_img(p_path, j_path, o_path, convt)
    
    
    return
'''
# Create angle vis in reflected image from laser scanner
def color_code_depth_img(ply_path, json_path, out_dir, convt):
    
    windowSize = 4
    xyScale = 3
    
    jsonf, plyf, gImf, pImf = find_files(ply_path, json_path)
    if jsonf == [] or plyf == [] or gImf == [] or pImf == []:
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    metadata = terra_common.lower_keys(terra_common.load_json(jsonf))
    yOffset = get_offset_from_metadata(metadata)
    
    gIm = Image.open(gImf)
    
    [gWid, gHei] = gIm.size
    
    # color code gray image
    codeImg = Image.new("RGB", gIm.size, "black")
    
    pix = np.array(gIm).ravel()
    
    gIndex = (np.where(pix>32))
    nonZeroSize = gIndex[0].size
    
    plydata = PlyData.read(plyf)
    
    pointSize = plydata.elements[0].count
    
    if nonZeroSize != pointSize:
        print('Point counts do not match.')
        return
    
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[gIndex[0]] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    angle_data = []
    for i in range(0,32):
        angle_data.append(np.zeros((1,6)))
        
    icount = 0
    # get top angle
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        icount = icount+1
        jcount = 0
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            jcount = jcount + 1
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plydata.elements[0].data[localIndex-1]
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(yCoord, yOffset, convt) - 1
            localNormal = calcAreaNormalSurface(localP)
            if localNormal != [] :
                angle_data[area_ind] = np.append(angle_data[area_ind],[localNormal], axis = 0)
    
    hist_data = np.zeros((32, 90))
    pix_height = np.zeros(32)
    disp_window = np.zeros(32)
    min_z  = np.zeros(32)
    ind = 0
    for meta_angle in angle_data:
        if meta_angle.size < 10:
            continue
        
        pix_height[ind] = get_scanned_height(meta_angle)
        leaf_angle = remove_soil_points(meta_angle)
        hist_data[ind] = gen_angle_hist_from_raw(meta_angle)
        disp_window[ind] = np.argmax(hist_data[ind])
        min_z[ind] = np.amin(meta_angle[1:,5])+55
        ind = ind + 1
    
    # color code
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        icount = icount+1
        jcount = 0
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            jcount = jcount + 1
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plydata.elements[0].data[localIndex-1]
            localNormal = calcAreaNormalSurface(localP)
            
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(yCoord, yOffset, convt) - 1
            if localNormal == [] :
                continue
            #if localNormal[5] < min_z[area_ind]:
            #    continue
            #if angle_in_range(disp_window[area_ind], localNormal):
            rgb = normals_to_rgb_2(localNormal)
            codeImg.paste(rgb, (i-windowSize*xyScale, j-windowSize, i+windowSize*xyScale+1, j+windowSize+1))
    
            #save_points(localP, '/Users/nijiang/Desktop/normal_plot.png', 4)
    
    img1 = Image.open(gImf)
    img1 = img1.convert('RGB')
    
    img3 = Image.blend(img1, codeImg, 0.5)
    save_png_file = os.path.join(out_dir, os.path.basename(gImf))
    img3.save(save_png_file)
    
    file_ind = 0
    for save_data in angle_data:
        file_ind = file_ind + 1
        out_basename = str(file_ind) + '.npy'
        out_file = os.path.join(out_dir, out_basename)
        np.save(out_file, save_data)
    
    json_basename = os.path.basename(jsonf)
    json_dst = os.path.join(out_dir, json_basename)
    shutil.copyfile(jsonf, json_dst)
    return
'''
def angle_in_range(disp_window, normals):
    
    min_angle = disp_window-3
    min_angle = min_angle if min_angle > 0 else 0
    max_angle = disp_window+3
    max_angle = max_angle if max_angle < 90 else 90
    #state = "fat" if is_fat else "not fat"
    
    r_min = math.cos(math.radians(max_angle))
    r_max = math.cos(math.radians(min_angle))
    
    if normals[2] > r_min and normals[2] < r_max:
        return True
    
    return False

def normals_to_rgb(normals):
    
    h = math.acos(normals[0])/math.pi
    s = math.acos(normals[1])/math.pi
    v = math.acos(normals[2])/math.pi
    rgb = colorsys.hsv_to_rgb(h, s, v)
    r = int(round(rgb[0]*255))
    g = int(round(rgb[1]*255))
    b = int(round(rgb[2]*255))
    
    return (r,g,b)

def normals_to_rgb_2(normals):
    
    val = math.atan2(normals[1], normals[0])
    
    h = (val+(0.5)*math.pi)/(math.pi)
    s = math.acos(normals[2])/(math.pi)
    v = 0.7
    rgb = colorsys.hsv_to_rgb(h, s, v)
    r = int(round(rgb[0]*255))
    g = int(round(rgb[1]*255))
    b = int(round(rgb[2]*255))
    
    return (r,g,b)

def convert_to_rgb(minval, maxval, val, colors):
    max_index = len(colors)-1
    v = float(val-minval) / float(maxval-minval) * max_index
    i1, i2 = int(v), min(int(v)+1, max_index)
    (r1, g1, b1), (r2, g2, b2) = colors[i1], colors[i2]
    f = v - i1
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def create_angle_data(str_date, ply_path, json_path, out_dir, convt):
    
    print("Start processing "+ os.path.basename(os.path.basename(ply_path)))
    
    windowSize = 4
    xyScale = 3
    
    # fetch input files from dirs
    #jsonf, plyf, gImf, pImf = find_files(ply_path, json_path)
    #if jsonf == [] or plyf == [] or gImf == [] or pImf == []:
    #    return
    files = find_files(ply_path, json_path)
    if files == None:
        return
    
    jsonf, plyf, gImf, pImf = files
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # if json file in dist dir, it's been processed, we skip it.
    json_basename = os.path.basename(jsonf)
    json_dst = os.path.join(out_dir, json_basename)
    #if os.path.exists(json_dst):
    #    return
    
    # obtain scan position and offsets from metadata
    metadata = lower_keys(load_json(jsonf))
    center_position = get_position(metadata)
    if center_position == None:
        return
    yOffset = get_offset_from_metadata(metadata)
    if yOffset == None:
        return
    
    plyFile = PlyData.read(plyf)
    plyData = plyFile.elements[0]
    
    # get relationship between ply files and png files, that means each point in the ply file 
    # should have a corresponding pixel in png files, both depth png and gray png
    pImg = cv2.imread(pImf, -1)
    gImg = cv2.imread(gImf, -1)
    pHei, pWid = pImg.shape[:2]
    gHei, gWid = gImg.shape[:2]
    if pWid == gWid:
        gPix = np.array(gImg).ravel()
        gIndex = (np.where(gPix>32))
        tInd = gIndex[0]
    else:
        pPix = np.array(pImg)
        pPix = pPix[:, 2:].ravel()
        pIndex = (np.where(pPix != 0))
        
        gPix = np.array(gImg).ravel()
        gIndex = (np.where(gPix>33))
        tInd = np.intersect1d(gIndex[0], pIndex[0])
                          
    nonZeroSize = tInd.size
    
    pointSize = plyData.count
    
    # if point size do not match, return
    if nonZeroSize != pointSize:
        return
    
    # Initial data structures
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[tInd] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    angle_data = []
    
    PLOT_COL_NUM = convt.max_col
    
    for i in range(0,PLOT_COL_NUM):
        angle_data.append(np.zeros((1,6)))
    
    '''
    (fields, traits) = get_traits_table_leafNormal()
    out_csv_path = os.path.join(out_dir, json_basename[:-4]+'csv')
    csvHandle = open(out_csv_path, 'w')
    csvHandle.write(','.join(map(str, fields)) + '\n')
    '''
    # move ROI in a window size to do the meta analysis
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            # fetch points in the ROI
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 50:
                continue
            localP = plyData.data[localIndex-1]
            xCoord = np.mean(localP["x"])
            yCoord = np.mean(localP["y"])
            
            '''
            sitename = get_sitename_from_3dCoordinates(xCoord, yCoord, yOffset, center_position[0], str_date)
            
            if len(sitename) == 0:
                continue
            
            localNormal = calcAreaNormalSurface(localP)
            if localNormal != [] :
                # data to trait list
                traits['date'] = str_date
                traits['site'] = sitename[0]['sitename']
                traits['normal0'] = localNormal[0]
                traits['normal1'] = localNormal[1]
                traits['normal2'] = localNormal[2]
                traits['x'] = localNormal[3]
                traits['y'] = localNormal[4]
                traits['z'] = localNormal[5]
                trait_list = generate_traits_list_leafNormal(traits)
                csvHandle.write(','.join(map(str, trait_list)) + '\n')
            
            '''
            
            # get ROI column number(32 columns in total)
            area_ind = get_area_index(xCoord, yCoord, yOffset, center_position[0], convt) - 1
            if area_ind < 0:
                continue
            # calculate angle data and store point position in localNormal
            localNormal = calcAreaNormalSurface(localP)
            if localNormal != [] :
                angle_data[area_ind] = np.append(angle_data[area_ind],[localNormal], axis = 0)
    
    file_ind = 0
    # save output data in npy files
    for save_data in angle_data:
        file_ind = file_ind + 1
        out_basename = str(file_ind) + '.npy'
        out_file = os.path.join(out_dir, out_basename)
        np.save(out_file, save_data)
    
    #csvHandle.close()
    shutil.copyfile(jsonf, json_dst)
    
    return

def generate_traits_list_leafNormal(traits):
    # compose the summary traits
    trait_list = [  traits['date'],
                    traits['site'],
                    traits['normal0'],
                    traits['normal1'],
                    traits['normal2'],
                    traits['x'],
                    traits['y'],
                    traits['z']
                ]

    return trait_list

def get_traits_table_leafNormal():

    fields = ('date','site', 'normal0', 'normal1', 'normal2', 'x', 'y', 'z')
    traits = {'date' : '',
              'site' : [],
              'normal0': [],
              'normal1': [],
              'normal2': [],
              'x': [],
              'y': [],
              'z': []}

    return (fields, traits)

def full_day_summary_from_csv(in_dir, out_dir):
    
    print(in_dir)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    dir_path = os.path.dirname(in_dir)
    
    str_date = os.path.basename(in_dir)
    # t0, first half of night, 18:00 to 24:00, t1, second day, second half of night, 0:00 to 8:00
    t1 = datetime.strptime(str_date, '%Y-%m-%d').date()
    t0 = t1+timedelta(-1)
    
    # collect target dirs in day t0 and t1
    target_dirs = []
    t0_dir = os.path.join(dir_path, str(t0))
    if os.path.isdir(t0_dir):
        list_t0_dir = os.listdir(t0_dir)
        for d in list_t0_dir:
            t0_sub_dir = os.path.join(t0_dir, d)
            if not os.path.isdir(t0_sub_dir):
                continue
            
            # t0 target hours from 18 to 24
            str_hour = d[12:14]
            date_hour = int(str_hour)
            if not date_hour in range(18, 24):
                continue
            target_dirs.append(t0_sub_dir)
    
    t1_dir = os.path.join(dir_path, str(t1))
    if os.path.isdir(t1_dir):
        list_t1_dir = os.listdir(t1_dir)
        for d in list_t1_dir:
            t1_sub_dir = os.path.join(t1_dir, d)
            if not os.path.isdir(t1_sub_dir):
                continue
            
            # t0 target hours from 18 to 24
            str_hour = d[12:14]
            date_hour = int(str_hour)
            if not date_hour in range(0, 8):
                continue
            target_dirs.append(t1_sub_dir)
    
    return

def full_day_summary(in_dir, out_dir, convt, dap):
    
    print(in_dir)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    dir_path = os.path.dirname(in_dir)
    
    str_date = os.path.basename(in_dir)
    # t0, first half of night, 18:00 to 24:00, t1, second day, second half of night, 0:00 to 8:00
    t1 = datetime.strptime(str_date, '%Y-%m-%d').date()
    t0 = t1+timedelta(-1)
    
    # collect target dirs in day t0 and t1
    target_dirs = []
    t0_dir = os.path.join(dir_path, str(t0))
    if os.path.isdir(t0_dir):
        list_t0_dir = os.listdir(t0_dir)
        for d in list_t0_dir:
            t0_sub_dir = os.path.join(t0_dir, d)
            if not os.path.isdir(t0_sub_dir):
                continue
            
            # t0 target hours from 18 to 24
            str_hour = d[12:14]
            date_hour = int(str_hour)
            if not date_hour in range(18, 24):
                continue
            target_dirs.append(t0_sub_dir)
    
    t1_dir = os.path.join(dir_path, str(t1))
    if os.path.isdir(t1_dir):
        list_t1_dir = os.listdir(t1_dir)
        for d in list_t1_dir:
            t1_sub_dir = os.path.join(t1_dir, d)
            if not os.path.isdir(t1_sub_dir):
                continue
            
            # t0 target hours from 18 to 24
            str_hour = d[12:14]
            date_hour = int(str_hour)
            if not date_hour in range(0, 8):
                continue
            target_dirs.append(t1_sub_dir)
        
    PLOT_RANGE_NUM = convt.max_range
    PLOT_COL_NUM = convt.max_col
    angleHist = np.zeros((PLOT_RANGE_NUM, PLOT_COL_NUM, 90))
    
    for d in target_dirs:
        full_path = os.path.join(in_dir, d)
        if not os.path.isdir(full_path):
            continue
            
        # fetch angle metadata from previous outputs
        fRange, hist_data, pix_height = get_angle_data(full_path, convt, dap)
            
        if fRange == None:
            continue
        
        # add it into main structure
        for j in range(0,PLOT_COL_NUM):
            angleHist[fRange-1, j] = angleHist[fRange-1, j]+hist_data[j]
    
    print('start saving')
    # save histogram to next level outputs        
    hist_out_csv = os.path.join(out_dir, 'angleHist.csv')
    outAngleHist = np.zeros((PLOT_RANGE_NUM*PLOT_COL_NUM, 92))
    outInd = 0
    for i in range(PLOT_RANGE_NUM):
        for j in range(PLOT_COL_NUM):
            outAngleHist[outInd][0] = i+1
            outAngleHist[outInd][1] = j+1
            outAngleHist[outInd][2:] = angleHist[i,j]
            outInd += 1
    np.savetxt(hist_out_csv, outAngleHist, delimiter="\t")
    
    relHist = np.zeros((PLOT_RANGE_NUM*PLOT_COL_NUM, 6))
    chiHist = np.zeros((PLOT_RANGE_NUM*PLOT_COL_NUM, 4))
    # generate beta fit and chi fit
    outInd = 0
    for i in range(PLOT_RANGE_NUM):
        for j in range(PLOT_COL_NUM):
            relHist[outInd][0] = i+1
            relHist[outInd][1] = j+1
            relHist[outInd][2:] = calc_beta_distribution_value(angleHist[i,j])
            
            chiHist[outInd][0] = i+1
            chiHist[outInd][1] = j+1
            chiHist[outInd][2:] = calc_chi_fit_parameter(angleHist[i,j])
            
            outInd += 1
                
    chiHistHandle = os.path.join(out_dir, 'beta-distribution.csv')
    np.savetxt(chiHistHandle, relHist, delimiter="\t")
    chiHistHandle = os.path.join(out_dir, 'chi-distribution.csv')
    np.savetxt(chiHistHandle, chiHist, delimiter="\t")
    
    return

def angleHist_to_BETY_Csv(in_dir, out_dir):
    
    d1 = date(2018, 5, 1)  # start date
    d2 = date(2018, 8, 1)  # end date

    delta = d2 - d1         # timedelta
    
    seasonNum = 6
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        print(str_date)
        
        sub_dir = os.path.join(in_dir, str_date)
        if not os.path.isdir(sub_dir):
            continue
        
        compute_output_from_angleHist(sub_dir, str_date, out_dir, seasonNum)
    
    return

def compute_output_from_angleHist(in_dir, str_date, out_dir, seasonNum, convt):
    
    PLOT_RANGE_NUM = convt.max_range
    PLOT_COL_NUM = convt.max_col
    
    hist_csv = os.path.join(in_dir, 'angleHist.csv')
    
    angleHist = np.loadtxt(hist_csv, delimiter='\t')
    '''
    plotAngleHist = np.zeros((864, 92))
    outInd = 0
    # combine 2 subplot into one
    for i in range(PLOT_RANGE_NUM):
        for j in range(16):
            plotAngleHist[outInd][0] = i+1
            plotAngleHist[outInd][1] = j+1
            h1 = angleHist[2*outInd][2:]
            h2 = angleHist[2*outInd+1][2:]
            targetHist = h1+h2
            plotAngleHist[outInd][2:] = targetHist
    '''
    # compute parameters and save to csv
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    out_file = os.path.join(out_dir, str_date+'_betaD.csv')
    csv = open(out_file, 'w')
    
    (fields, traits) = get_traits_table_leafAngle()
        
    csv.write(','.join(map(str, fields)) + '\n')
    
    for i in range(PLOT_RANGE_NUM):
        for j in range(PLOT_COL_NUM):
            ind = i*PLOT_COL_NUM+j
            fRange = angleHist[ind][0]
            fCol = angleHist[ind][1]
            targetHist = angleHist[ind][2:]
            if (targetHist.max() == 0):
                continue
            else: 
                rel = calc_beta_distribution_value(targetHist)
                if rel == None:
                    continue
                
                chi = calc_chi_fit_parameter(targetHist)
                if rel == None:
                    continue
                
                str_time = str_date+'T12:00:00'
                traits['local_datetime'] = str_time
                traits['leaf_angle_mean'] = rel[0]
                traits['leaf_angle_variance'] = rel[1]
                traits['leaf_angle_alpha'] = rel[2]
                traits['leaf_angle_beta'] = rel[3]
                traits['leaf_angle_chi'] = chi[1]
                traits['site'] = parse_site_from_range_column(fRange, fCol, seasonNum)
                trait_list = generate_traits_list(traits)
                csv.write(','.join(map(str, trait_list)) + '\n')
        

    csv.close()
    
    return

def remove_ground_aggregator(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_hist_dir = os.path.join(out_dir, 'Hist')
    if not os.path.isdir(out_hist_dir):
        os.makedirs(out_hist_dir)
        
    out_bety_dir = os.path.join(out_dir, 'Bety')
    if not os.path.isdir(out_bety_dir):
        os.makedirs(out_bety_dir)
    
    
    d1 = date(2017, 5, 1)  # start date
    d2 = date(2017, 9, 1)  # end date

    delta = d2 - d1         # timedelta
    
    (fields, traits) = get_traits_table_leafAngle()
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        
        print(str_date)
        
        dir_path = in_dir
    
        # t0, first half of night, 18:00 to 24:00, t1, second day, second half of night, 0:00 to 8:00
        t1 = datetime.strptime(str_date, '%Y-%m-%d').date()
        t0 = t1+timedelta(-1)
        
        # collect target dirs in day t0 and t1
        target_dirs = []
        t0_dir = os.path.join(dir_path, str(t0))
        if os.path.isdir(t0_dir):
            list_t0_dir = os.listdir(t0_dir)
            for d in list_t0_dir:
                t0_sub_dir = os.path.join(t0_dir, d)
                if not os.path.isdir(t0_sub_dir):
                    continue
                
                # t0 target hours from 18 to 24
                str_hour = d[12:14]
                date_hour = int(str_hour)
                if not date_hour in range(18, 24):
                    continue
                target_dirs.append(t0_sub_dir)
        
        t1_dir = os.path.join(dir_path, str(t1))
        if os.path.isdir(t1_dir):
            list_t1_dir = os.listdir(t1_dir)
            for d in list_t1_dir:
                t1_sub_dir = os.path.join(t1_dir, d)
                if not os.path.isdir(t1_sub_dir):
                    continue
                
                # t0 target hours from 18 to 24
                str_hour = d[12:14]
                date_hour = int(str_hour)
                if not date_hour in range(0, 8):
                    continue
                target_dirs.append(t1_sub_dir)
                
        angleHist = np.zeros((PLOT_RANGE_NUM, 16, 90))
        
        if len(target_dirs) == 0:
            continue
        
        convt = terra_common.CoordinateConverter()
        q_flag = convt.bety_query(str_date, True)
        if not q_flag:
            print('Bety query for boundaries failed')
            continue
    
        for d in target_dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
                
            # fetch angle metadata from previous outputs
            fRange, hist_data = fetch_angle_data_remove_ground(full_path, convt, i)
                
            if fRange == None:
                continue
            
            # add it into main structure
            for j in range(0,16):
                angleHist[fRange-1, j] = angleHist[fRange-1, j]+hist_data[j]
                
        print('start saving')
        # save histogram to next level outputs  
        out_path = os.path.join(out_hist_dir, str_date)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        hist_out_csv = os.path.join(out_path, 'angleHist.csv')
        outAngleHist = np.zeros((PLOT_RANGE_NUM*16, 92))
        outInd = 0
        for i in range(PLOT_RANGE_NUM):
            for j in range(16):
                outAngleHist[outInd][0] = i+1
                outAngleHist[outInd][1] = j+1
                outAngleHist[outInd][2:] = angleHist[i,j]
                outInd += 1
        np.savetxt(hist_out_csv, outAngleHist, delimiter="\t")
        
        relHist = np.zeros((PLOT_RANGE_NUM*16, 6))
        chiHist = np.zeros((PLOT_RANGE_NUM*16, 4))
        # generate beta fit and chi fit
        outInd = 0
        for i in range(PLOT_RANGE_NUM):
            for j in range(16):
                relHist[outInd][0] = i+1
                relHist[outInd][1] = j+1
                relHist[outInd][2:] = calc_beta_distribution_value(angleHist[i,j])
                
                chiHist[outInd][0] = i+1
                chiHist[outInd][1] = j+1
                chiHist[outInd][2:] = calc_chi_fit_parameter(angleHist[i,j])
                
                outInd += 1
                    
        chiHistHandle = os.path.join(out_path, 'beta-distribution.csv')
        np.savetxt(chiHistHandle, relHist, delimiter="\t")
        chiHistHandle = os.path.join(out_path, 'chi-distribution.csv')
        np.savetxt(chiHistHandle, chiHist, delimiter="\t")
        
        insert_leafAngle_traits_into_betydb(out_path, out_bety_dir, str_date, convt)
    
    return

def load_chi_beta_explort_to_betyCsv(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_bety_dir = os.path.join(out_dir, 'BETY')
    if not os.path.isdir(out_bety_dir):
        os.makedirs(out_bety_dir)
        
    
    out_hist_dir = os.path.join(out_dir, 'hist')
    if not os.path.isdir(out_hist_dir):
        os.makedirs(out_hist_dir)
    
    
    d1 = date(2017, 5, 1)  # start date
    d2 = date(2017, 9, 1)  # end date

    delta = d2 - d1         # timedelta
    
    (fields, traits) = get_traits_table_leafAngle()
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        chi_file = 'chi-distribution.csv'
        chi_path = os.path.join(in_dir, str_date, chi_file)
        if not os.path.isfile(chi_path):
            continue
        
        beta_file = 'beta-distribution.csv'
        beta_path = os.path.join(in_dir, str_date, beta_file)
        if not os.path.isfile(beta_path):
            continue
        
        chiHist = np.loadtxt(open(chi_path, 'r'), delimiter='\t')
        betaHist = np.loadtxt(open(beta_path, 'r'), delimiter='\t')
        
        out_csv_path = os.path.join(out_bety_dir, '{}_leafAngle.csv'.format(str_date))
        csvHandle = open(out_csv_path, 'w')
        csvHandle.write(','.join(map(str, fields)) + '\n')
        
        for j in range(len(chiHist)):
            out_chi_line = chiHist[j]
            out_beta_line = betaHist[j]
            
            if out_chi_line[2] < 0 or out_chi_line[2] > 5:
                chiHist[j][2:] = -1
                betaHist[j][2:] = -1
                continue
            
            str_time = str_date+'T12:00:00'
            traits['local_datetime'] = str_time
            traits['leaf_angle_alpha_src'] = out_beta_line[2]
            traits['leaf_angle_beta_src'] = out_beta_line[3]
            traits['leaf_angle_alpha_fit'] = out_beta_line[4]
            traits['leaf_angle_beta_fit'] = out_beta_line[5]
            traits['leaf_chi_src'] = out_chi_line[2]
            traits['leaf_chi_fit'] = out_chi_line[3]
            traits['site'] = parse_site_from_range_column(out_chi_line[0], out_chi_line[1], 4)
            trait_list = generate_traits_list(traits)
            csvHandle.write(','.join(map(str, trait_list)) + '\n')
            
        csvHandle.close()
        
        chiHistHandle = os.path.join(out_hist_dir, '{}_chi.csv'.format(str_date))
        np.savetxt(chiHistHandle, chiHist, delimiter="\t")
        
    
    return

def parse_site_from_range_column(row, col, seasonNum):
    
    rel = 'MAC Field Scanner Season {} Range {} Column {}'.format(str(seasonNum), str(int(row)), str(int(col)))
    
    return rel

def fetch_angle_data_remove_ground(in_dir, convt, dap):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
        
    hist_data = np.zeros((16,90))
    # parse json file
    metafile, angle_files = find_result_files(in_dir)
    if metafile == [] or angle_files == [] :
        print('No meta file or angle files')
        return None, None
    
    metadata = terra_common.lower_keys(terra_common.load_json(metafile))
    center_position = get_position(metadata)
    
    fRange = field_2_range(center_position[0], convt)
    if fRange == 0:
        return None, None
    
    for i in range(0,16):
        file_path = os.path.join(in_dir, str(i*2+1)+'.npy')
        if not os.path.exists(file_path):
            continue
        
        meta_angle = np.load(file_path, 'r')
    
        if len(meta_angle) < 300:
            continue
        
        leaf_angle = remove_soil_points(meta_angle, dap)
        
        if len(leaf_angle) < 300:
            continue
        
        hist_data[i] = gen_angle_hist_from_raw(leaf_angle)
    
    return fRange, hist_data

def get_angle_data(in_dir, convt, dap):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
        return
    
    PLOT_RANGE_NUM = convt.max_range
    PLOT_COL_NUM = convt.max_col
        
    hist_data = np.zeros((PLOT_COL_NUM,90))
    pix_height = np.zeros(PLOT_COL_NUM)
    # parse json file
    metafile, angle_files = find_result_files(in_dir)
    if metafile == [] or angle_files == [] :
        print('No meta file or angle files')
        return None, None, None
    
    metadata = terra_common.lower_keys(terra_common.load_json(metafile))
    center_position = get_position(metadata)
    
    fRange = field_2_range(center_position[0], convt)
    if fRange == 0:
        return None, None, None
    
    for i in range(0,PLOT_COL_NUM):
        file_path = os.path.join(in_dir, str(i+1)+'.npy')
        if not os.path.exists(file_path):
            continue
        
        meta_angle = np.load(file_path, 'r')
        
        #out_file = os.path.join(in_dir, str(i+1)+'.png')
        #create_angle_visualization(meta_angle, out_file)
        if meta_angle.size < 10:
            continue
        
        #pix_height[i] = get_scanned_height(meta_angle)
        
        leaf_angle = remove_soil_points(meta_angle, dap)
        
        hist_data[i] = gen_angle_hist_from_raw(leaf_angle)
    
    return fRange, hist_data, pix_height

from skimage import filters

def remove_soil_points(meta_angle, dap=0):
    
    MIN_HEIGHT = 100
    point_z = meta_angle[1:,5]
    
    if dap < 20:
        z_hist, bounds = np.histogram(point_z, bins=100)
        max_ind = np.argmax(z_hist)
        ground_level = bounds[max_ind]
    elif dap > 40:
        ground_level = np.amin(point_z)
    else:
        ground_level = filters.threshold_otsu(point_z, nbins = 100)
    
    ground_level += MIN_HEIGHT
    
    specifiedIndex = np.where(meta_angle[:,5]>ground_level)
    target = meta_angle[specifiedIndex]
    
    return target



# get initial soil height for season 2
def get_scanned_height(meta_angle):
    
    point_z = meta_angle[1:,5]
    pix_min = np.amin(point_z)
    pix_max = np.amax(point_z)
    
    return pix_max-pix_min

def create_angle_visualization(dataset, out_file):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #colors = cm.rainbow(np.linspace(0, 1, 32))
    #ax.scatter(X,Y,Z,color=colors[5], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    for metadata in dataset:
        if metadata[0] == 0:
            continue
        ax.quiver(metadata[3], metadata[4], metadata[5], metadata[0], metadata[1], metadata[2], length=15)
    
    plt.draw()
    plt.savefig(out_file)
    plt.close()
    
    
    return

def gen_angle_hist_from_raw(raw_data):
    
    plot_angle_hist = np.zeros(90)
    
    zVal = raw_data[:, 2]
    
    for i in range(0, 90):
        r_max = math.cos(math.radians(i))
        r_min = math.cos(math.radians(i+1))
        
        histObj = np.where(np.logical_and(zVal > r_min, zVal < r_max))
        
        plot_angle_hist[i] = histObj[0].size
        
    return plot_angle_hist

def field_2_range(x_position, convt):

    xRange = 0
    PLOT_RANGE_NUM = convt.max_range
    
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (x_position > xmin) and (x_position <= xmax):
            xRange = i + 1
            return xRange
    
    return 0


def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        
        camera_x = sensor_fix_meta['scanner west location in camera box x [m]']

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        return None

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z)
    except ValueError as err:
        fail('Corrupt positions, ' + err.args[0])
        return None
    return (x, y, z)

def find_result_files(in_dir):
    
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print (in_dir)
        fail('Could not find .json file')
        return [], []
    
    npy_suffix = os.path.join(in_dir, '*.npy')
    npys = glob(npy_suffix)
    if len(npys) == 0:
        fail('Could not find .npy files')
        return [], []
    
    return jsons[0], npys
    

def get_offset_from_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        return None
        
    if scan_direction == 'False':
        yShift = -25.711
    else:
        yShift = -3.60
    
    return yShift

def get_area_index(xCoord, yCoord, yOffset, gantry_x, convt):
    
    PLOT_RANGE_NUM = convt.max_range
    PLOT_COL_NUM = convt.max_col
    
    xPosition = xCoord/1000 + gantry_x
    yPosition = yCoord/1000 - yOffset
    
    count = -1
    
    xRange = 0
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (xPosition > xmin) and (xPosition <= xmax):
            xRange = i + 1
            break
            
    if xRange == 0:
        return count
        
    for j in range(PLOT_COL_NUM):
        ymin = convt.np_bounds[xRange-1][j][2]
        ymax = convt.np_bounds[xRange-1][j][3]
        if (yPosition > ymin) and (yPosition <= ymax):
            count = j + 1
            break
    
    return count

# Campbell distribution function
def chi_distribution(t, init_chi, leaf_lambda):
    
    t = t * math.pi/2
    
    a = 2 * (init_chi ** 3) * np.sin(t)
    b = leaf_lambda * (np.cos(t)**2 + (init_chi**2) *(np.sin(t)**2))**2
    
    ret = a/b
    
    return ret


# beta distribution function
def beta_distribution(t, tMean, tVar):
    
    delta0 = tMean*(1-tMean)
    deltaT = tVar
    
    u = (1-tMean)*(delta0/deltaT-1)
    v = tMean*(delta0/deltaT-1)
    
    if u < 0 or v < 0 :
        return -100
    
    try:
        B = math.gamma(u)*math.gamma(v)/math.gamma(u+v)
    except OverflowError:
        B = float('inf')

    #B = math.gamma(u)*math.gamma(v)/math.gamma(u+v)
    
    a = np.power(1-t,u-1)
    b = np.power(t, v-1)
    c = a * b
    ret = c/B
    
    return ret

def beta_mu_nu(tMean, tVar):
    
    delta0 = tMean*(1-tMean)
    deltaT = tVar
    
    u = (1-tMean)*(delta0/deltaT-1)
    v = tMean*(delta0/deltaT-1)
    
    return u, v

def get_mean_and_variance_from_angleHist(angleHist):
    
    tSum = 0
    for i in range(angleHist.size):
        tSum += (i/90.0) * angleHist[i]
    tMean = tSum
    
    tVar = 0
    for i in range(angleHist.size):
        tVar += (i/90.0-tMean)*(i/90.0-tMean)*angleHist[i]
    
    return tMean, tVar

def calc_init_chi(tMean):
    
    tMean = tMean * math.pi/2
    
    init_chi = -3 + pow(tMean/9.65, -0.6061)
    
    # not clear how to compute leaf lambda if init_chi > 1, use the approximate equation
    # https://github.com/terraref/computing-pipeline/issues/338#issuecomment-357271181
    leaf_lambda = init_chi + 1.774*pow(init_chi+1.182, -0.733)
    '''
    eata = pow(1-init_chi*init_chi, 0.5)
    
    if init_chi < 1:
        leaf_lambda = init_chi + math.asin(eata)/eata
    else:
        leaf_lambda = init_chi + 
    '''
    
    return init_chi, leaf_lambda

def calc_chi_fit_parameter(np_x):
    
    if np.amax(np_x) < 5:
        return
    
    if np.isnan(np.min(np_x)):
        return
    
    angleHist = np_x / np.sum(np_x)
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    init_chi, leaf_lambda = calc_init_chi(tMean)
    x = linspace(0.01, 0.99, 90)
    y = angleHist
    delta_y = 90
    y = y * delta_y
    gmod = Model(chi_distribution)
    try:
        result = gmod.fit(y, t=x, init_chi = init_chi, leaf_lambda = leaf_lambda)
    except ValueError as err:
        print(err.args[0])
    
    rel = []
    rel.append(init_chi)
    rel.append(result.best_values['init_chi'])
    
    return rel
    
    
def calc_beta_distribution_value(np_x):
    
    if np.amax(np_x) < 5:
        return
    
    if np.isnan(np.min(np_x)):
        return
    
    angleHist = np_x / np.sum(np_x)
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    x = linspace(0.01, 0.99, 90)
    y = angleHist
    delta_y = 90
    y = y * delta_y
    gmod = Model(beta_distribution)
    try:
        result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
    except ValueError as err:
        print(err.args[0])
        return
    
    rel = []
    rel.append(tMean) 
    rel.append(math.sqrt(tVar))
    a, b = beta_mu_nu(result.best_values['tMean'], result.best_values['tVar'])
    rel.append(a)
    rel.append(b)
    
    return rel

def load_leaf_angle_parameter(in_dir):
    
    betaNpyHandle = os.path.join(in_dir, 'beta-distribution.csv')
    if not os.path.exists(betaNpyHandle):
        return []
    relHist = np.genfromtxt(betaNpyHandle, delimiter='\t')
    
    chiNpyHandle = os.path.join(in_dir, 'chi-distribution.csv')
    if not os.path.exists(chiNpyHandle):
        return []
    chiHist = np.genfromtxt(chiNpyHandle, delimiter='\t')
    
    return relHist, chiHist

def get_traits_table_leafAngle():
    
    fields = ('local_datetime', 'access_level', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method',
              'leaf_angle_mean', 'leaf_angle_variance', 'leaf_angle_alpha', 'leaf_angle_beta',
              'leaf_angle_chi')
    traits = {'local_datetime' : '',
              'access_level': '2',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2018',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': '3D scanner to leaf angle distribution',
              'leaf_angle_mean': [],
              'leaf_angle_variance': [],
              'leaf_angle_alpha': [],
              'leaf_angle_beta': [],
              'leaf_angle_chi': []}

    return (fields, traits)

def parse_site_from_range_col(fRange, fCol, convt):
    
    Column = (fCol+1)/2
    if (fCol % 2) != 0:
        subplot = 'W'
    else:
        subplot = 'E'
    
    seasonNum = convt.seasonNum
    
    rel = 'MAC Field Scanner Season {} Range {} Column {} {}'.format(str(seasonNum), str(int(fRange)), str(int(Column)), subplot)
    
    return rel
    

def parse_site_from_plotNum_1728(plotNum, convt):

    plot_row = 0
    plot_col = 0
    
    cols = 32
    col = (plotNum-1) % cols + 1
    row = (plotNum-1) / cols + 1
    
    
    if (row % 2) != 0:
        plot_col = col
    else:
        plot_col = cols - col + 1
    
    Range = row
    Column = (plot_col + 1) / 2
    if (plot_col % 2) != 0:
        subplot = 'W'
    else:
        subplot = 'E'
        
    seasonNum = convt.seasonNum
        
    rel = 'MAC Field Scanner Season {} Range {} Column {} {}'.format(str(seasonNum), str(Range), str(Column), subplot)
    
    return rel

def generate_traits_list(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['access_level'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method'],
                    traits['leaf_angle_mean'],
                    traits['leaf_angle_variance'],
                    traits['leaf_angle_alpha'],
                    traits['leaf_angle_beta'],
                    traits['leaf_angle_chi']
                ]

    return trait_list

def insert_leafAngle_traits_into_betydb(in_dir, out_dir, str_date, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    hist, chi_hist = load_leaf_angle_parameter(in_dir)
    
    out_file = os.path.join(out_dir, str_date+'_betaD.csv')
    csv = open(out_file, 'w')
    
    (fields, traits) = get_traits_table_leafAngle()
        
    csv.write(','.join(map(str, fields)) + '\n')
    
    PLOT_RANGE_NUM = convt.max_range
    PLOT_COL_NUM = convt.max_col
    
    for i in range(PLOT_RANGE_NUM):
        for j in range(PLOT_COL_NUM):
            ind = i*PLOT_COL_NUM+j
            fRange = hist[ind][0]
            fCol = hist[ind][1]
            targetHist = hist[ind][2:]
            chiHist = chi_hist[ind][2:]
            if (targetHist.max() == -1):
                continue
            else: 
                str_time = str_date+'T12:00:00'
                traits['local_datetime'] = str_time
                traits['leaf_angle_alpha_src'] = targetHist[0]
                traits['leaf_angle_beta_src'] = targetHist[1]
                traits['leaf_angle_alpha_fit'] = targetHist[2]
                traits['leaf_angle_beta_fit'] = targetHist[3]
                traits['leaf_chi_src'] = chiHist[0]
                traits['leaf_chi_fit'] = chiHist[1]
                traits['site'] = parse_site_from_range_column(fRange, fCol, 4)
                trait_list = generate_traits_list(traits)
                csv.write(','.join(map(str, trait_list)) + '\n')
        

    csv.close()
    # switch to submit traits
    #betydb.submit_traits(out_file, filetype='csv', betykey=betydb.get_bety_key(), betyurl=betydb.get_bety_url())
    
    
    return

def draw_beta_distribution(in_dir, out_dir):
    
    filePath = os.path.join(in_dir, 'angleHist.npy')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    oneDayHist = np.load(filePath, 'r')
    
    for i in range(oneDayHist.size-1):
        if np.amax(oneDayHist[i]) < 5:
            continue
        
        if np.isnan(np.min(oneDayHist[i])):
            continue
    
        angleHist = oneDayHist[i] / np.sum(oneDayHist[i])
        
        tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
        
        x = linspace(0.01, 0.99, 90)
        y = angleHist
        delta_y = 90
        y = y * delta_y
        
        gmod = Model(beta_distribution)
        try:
            result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
        except ValueError as err:
            print(err.args[0])
        
        print(result.fit_report())
        
        plt.plot(x,y, 'bo')
        plt.plot(x, result.init_fit, 'k--')
        plt.plot(x, result.best_fit, 'r-')
        plt.title('plot num: %d' % (i+1))
        textline = 'reduced chi-square: %f\ndotted line: initial fit by beta distribution\nred line: best fitted\nblue plot: LAD from laser scanner'%(result.redchi)
        plt.annotate(textline, xy=(1, 1), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top')
        out_file = os.path.join(out_dir, str(i)+'.png')
        plt.savefig(out_file)
        plt.close()
    
    
    return

def draw_chi_distribution(in_dir, out_dir):
    
    filePath = os.path.join(in_dir, 'angleHist.npy')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    oneDayHist = np.load(filePath, 'r')
    
    for i in range(oneDayHist.size-1):
        if np.amax(oneDayHist[i]) < 5:
            continue
        
        if np.isnan(np.min(oneDayHist[i])):
            continue
    
        angleHist = oneDayHist[i] / np.sum(oneDayHist[i])
        
        tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
        init_chi, leaf_lambda = calc_init_chi(tMean)
        x = linspace(0.01, 0.99, 90)
        y = angleHist
        delta_y = 90
        y = y * delta_y
        gmod = Model(chi_distribution)
        try:
            result = gmod.fit(y, t=x, init_chi = init_chi, leaf_lambda = leaf_lambda)
        except ValueError as err:
            print(err.args[0])
        
        print(result.fit_report())
        
        plt.plot(x,y, 'bo')
        plt.plot(x, result.init_fit, 'k--')
        plt.plot(x, result.best_fit, 'r-')
        plt.title('plot num: %d' % (i+1))
        textline = 'dotted line: initial fit by chi-distribution\nred line: best fitted\nblue plot: LAD from laser scanner'
        plt.annotate(textline, xy=(1, 1), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top')
        out_file = os.path.join(out_dir, str(i)+'.png')
        plt.savefig(out_file)
        plt.close()
    
    
    return

def calcAreaNormalSurface(Q):
    
    nd3points = np.zeros((Q.size, 3))
    centerPoint = np.zeros(3)
    centerPoint[0] = np.mean(Q["x"])
    centerPoint[1] = np.mean(Q["y"])
    centerPoint[2] = np.mean(Q["z"])
    
    nd3points[:,0] = Q["x"] - np.mean(Q["x"])
    nd3points[:,1] = Q["y"] - np.mean(Q["y"])
    nd3points[:,2] = Q["z"] - np.mean(Q["z"])
    
    U, s, V = np.linalg.svd(nd3points, full_matrices=False)
    
    normals = V[2,:]
    normals = np.sign(normals[2])*normals
    
    s1 = s[0]/s[1]
    s2 = s[1]/s[2]
    #show_points_and_normals(nd3points, normals, centerPoint)
    if s1 < s2:
        reval = np.append(normals, centerPoint)
        return reval
    else:
        return []
    
    return normals

def show_points_and_normals(ply_data, normals, centerPoint):
    
    X = ply_data[:,0]
    Y = ply_data[:,1]
    Z = ply_data[:,2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = cm.rainbow(np.linspace(0, 1, 32))
    ax.scatter(X,Y,Z,color=colors[5], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.quiver(0, 0, 0, normals[0], normals[1], normals[2], length=2)
    
    plt.draw()
    #plt.savefig(out_file)
    plt.close()
    
    return

def find_files(ply_path, json_path):
    json_suffix = os.path.join(json_path, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print (json_path)
        fail('Could not find .json file')
        return None
    
    ply_suffix = os.path.join(ply_path, '*-west_0.ply')
    plys = glob(ply_suffix)
    if len(plys) == 0:
        print (ply_path)
        fail('Could not find west ply file')
        return None
    
    gIm_suffix = os.path.join(json_path, '*-west_0_g.png')
    gIms = glob(gIm_suffix)
    if len(gIms) == 0:
        fail('Could not find -west_0_g.png file')
        return None
    
    pIm_suffix = os.path.join(json_path, '*-west_0_p.png')
    pIms = glob(pIm_suffix)
    if len(pIms) == 0:
        fail('Could not find -west_0_p.png file')
        return None
    
    return jsons[0], plys[0], gIms[0], pIms[0]

def fail(reason):
    print >> sys.stderr, reason

def test():
    
    str_date = '2019-06-18'
    ply_path = '/media/zli/Elements/ua-mac/Level_1/scanner3DTop/2019-06-18/2019-06-18__03-20-38-542'
    json_path = '/media/zli/Elements/ua-mac/raw_data/scanner3DTop/2019-06-18/2019-06-18__03-20-38-542'
    out_dir = '/media/zli/Elements/ua-mac/Level_2/scanner3DTop/2019-06-18/2019-06-18__03-20-38-542'
    convt = None
    create_angle_data(str_date, ply_path, json_path, out_dir, convt)
    
    
    return

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
    
def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        return

def full_season_leafAngle_frame(ply_dir, json_dir, out_dir, save_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    print(deltaDay.days)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        ply_path = os.path.join(ply_dir, str_date)
        json_path = os.path.join(json_dir, str_date)
        out_path = os.path.join(out_dir, str_date)
        
        if not os.path.isdir(ply_path):
            continue
        
        if not os.path.isdir(json_path):
            continue
        
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        
        #full_day_gen_angle_multi_process(str_date, ply_path, json_path, out_path, convt)
        full_day_gen_angle_data(str_date, ply_path, json_path, out_path, convt)
    
    return


def full_season_leafAngle_integrate(in_dir, out_dir, bety_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        in_path = os.path.join(in_dir, str_date)
        out_path = os.path.join(out_dir, str_date)
        if not os.path.isdir(in_path):
            continue
        
        full_day_summary(in_path, out_path, convt, i)
        compute_output_from_angleHist(out_path, str_date, bety_dir, 9, convt)
    
    return

def main():
    
    print("start...")
    
    args = options()
    
    start_date = '2019-04-18'
    end_date = '2019-08-31'
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2019-06-18') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    
    full_season_leafAngle_frame(args.ply_dir, args.json_dir, args.out_dir, args.save_dir, start_date, end_date, convt)
    
    full_season_leafAngle_integrate(args.out_dir, args.save_dir, args.bety_dir, start_date, end_date, convt)
    
    return

if __name__ == "__main__":
    
    #load_chi_beta_explort_to_betyCsv('/media/zli/Elements/ua-mac/Level_2/leafAngle/Leaf_angle_s4', '/media/zli/Elements/ua-mac/Level_2/leafAngle/s4_betyCsv')
    
    # \remove_ground_aggregator('/home/zli/data/Terra/ua-mac/Level_2/leafAngleDistribution', '/home/zli/data/Terra/ua-mac/Level_2/leafAngleBetyS4')
    
    #angleHist_to_BETY_Csv('/media/zli/Elements/Terra_Bety_output/leafAngleBetyS6/s6_leaf_angle_Hist', '/media/zli/Elements/Terra_Bety_output/leafAngleBetyS6/LeafAngle_Bety_s6')
    
    main()
    #test()
    
    
    
    
    
    
    
    
    