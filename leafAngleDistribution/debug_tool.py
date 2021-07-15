'''
Created on May 9, 2019

@author: zli
'''

import os, sys, json, math, colorsys
import numpy as np
import terra_common
from datetime import date, timedelta
from glob import glob
from skimage import filters
from lmfit import Model
from plyfile import PlyData, PlyElement
from PIL import Image
import matplotlib.pyplot as plt
import cv2

PLOT_RANGE_NUM = 54
PLOT_COL_NUM = 32

def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        camera_x = '2.070'#sensor_fix_meta['scanner west location in camera box x [m]']
        

    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z)
    except ValueError as err:
        terra_common.fail('Corrupt positions, ' + err.args[0])
    return (x, y, z)

def get_direction(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])
        
    return scan_direction

def offset_choice(metadata):
    
    scanDirectory = get_direction(metadata)
    center_position = get_position(metadata)
    
    
    if scanDirectory == 'True':
        yShift = 3.45
    else:
        yShift = 25.711
        
    xShift = 0.082 + center_position[0]
    
    return xShift, yShift

def field_x_2_range(x_position, convt):
    
    xRange = 0
    
    for i in range(52):
        xmin = convt.np_bounds_subplot[i][0][0]
        xmax = convt.np_bounds_subplot[i][0][1]
        if (x_position > xmin) and (x_position < xmax):
            xRange = i + 1
    
    return xRange

def split_anotation_image(im2show, gImf, ply, metadata, pImf, out_dir, base_name, convt):
    
    # extract gantry position from json file
    center_position = get_position(metadata)
    
    # point cloud to gantry position offset, base on scan direction
    xShift, yShift = offset_choice(metadata)
    
    
    yRange = 32
    scaleParam = 1000
    
    # locate xRange #(1-54)
    xRange = field_x_2_range(center_position[0], convt)
    
    # load data
    ply_data = PlyData.read(ply)
    pImg = cv2.imread(pImf, -1)
    gImg = cv2.imread(gImf, -1)
    pHei, pWid = pImg.shape[:2]
    gHei, gWid = gImg.shape[:2]
    
    # get relationship between ply files and png files, that means each point in the ply file 
    # should have a corresponding pixel in png files, both depth png and gray png
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
        
    pointSize = ply_data.elements[0].count
    
    data = ply_data.elements[0].data
    
    # if point size does not match, return
    if nonZeroSize != pointSize:
        return []
    
    # Initial data structures
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[tInd] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    list_g_bounds = []
    
    # scan yRange(1-32)
    for i in range(yRange):
        ymin = (convt.np_bounds_subplot[xRange][i][2]-yShift) * scaleParam
        ymax = (convt.np_bounds_subplot[xRange][i][3]-yShift) * scaleParam
        
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        if len(target) == 0:
            continue
        
        # get y-axis boundaries in png file
        g_s_i = tInd[specifiedIndex[0]]
        g_start_row = min(g_s_i)/gWid
        g_end_row = max(g_s_i)/gWid
        
        # xRange, yRange to plotNumber
        plotNum = convt.fieldPartition_to_plotNum_32(xRange, i+1)
        
        if min(g_start_row,g_end_row) < 1 or max(g_start_row,g_end_row) > gHei or g_start_row == g_end_row:
            continue
        
        # crop png file
        crop_img = im2show[g_start_row:g_end_row, :]
        
        out_plot_dir = os.path.join(out_dir, str(plotNum))
        if not os.path.isdir(out_plot_dir):
            os.mkdir(out_plot_dir)
        
        out_file_path = os.path.join(out_plot_dir, '{}_{}.png'.format(base_name, int(center_position[0])))
        cv2.imwrite(out_file_path, crop_img)
        
        #plyIndices = gIndexImage_[int(box[1]):int(box[3]), int(box[0]):int(box[2])]    
    
    print(list_g_bounds)
    
    
    return

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
        
    # get top angle
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
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
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
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


def create_angle_data_debug(p_path, j_path, out_dir, str_date, convt, dap):
    
    windowSize = 4
    xyScale = 3
    
    jsonf, plyf, gImf, pImf = find_files(p_path, j_path)
    if jsonf == [] or plyf == [] or gImf == [] or pImf == []:
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    # obtain scan position and offsets from metadata
    metadata = lower_keys(load_json(jsonf))
    xShift, yShift = offset_choice(metadata)
    center_position = get_position(metadata)
    '''
    # save z position to file
    z_file = os.path.join(out_dir, 'gantryZ.txt')
    zHandle = open(z_file, 'a+')
    zHandle.write('%s,%f\n'%(str_date, center_position[2]))
    zHandle.close()
    return
    '''
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
    if nonZeroSize != pointSize or pointSize == 0:
        return
    
    # Initial data structures
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[tInd] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    groundLevel = estimate_ground_level(plyData.data, dap)
    
    # initial outputs
    angle_data = []
    for i in range(0,PLOT_COL_NUM):
        angle_data.append(np.zeros((1,6)))
        
    # color code gray image
    codeImg = Image.new("RGB", [gWid, gHei], "black")
    
    # comopute surface normal & color code reflectance
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
            plyIndices = plyIndices.ravel()
            plyIndices_ = np.where(plyIndices>0)
            
            localIndex = plyIndices[plyIndices_[0]].astype('int64')
            if plyIndices_[0].size < 100:
                continue
            localP = plyData.data[localIndex-1]
            zCoord = np.mean(localP['z'])
            if zCoord < groundLevel:
                continue
            
            localNormal = calcAreaNormalSurface(localP)
            xCoord = np.mean(localP["x"])
            yCoord = np.mean(localP["y"])
            area_ind = get_area_index(xCoord, yCoord, yShift, center_position[0], convt) - 1
            if localNormal == [] :
                continue
            
            rgb = normals_to_rgb_2(localNormal)
            codeImg.paste(rgb, (i-windowSize*xyScale, j-windowSize, i+windowSize*xyScale+1, j+windowSize+1))
            
            angle_data[area_ind] = np.append(angle_data[area_ind],[localNormal], axis = 0)
    
    yRange = PLOT_COL_NUM
    scaleParam = 1000
    
    img1 = Image.open(gImf)
    img1 = img1.convert('RGB')
    
    img3 = Image.blend(img1, codeImg, 0.5)
    temp_img_file = os.path.join(out_dir, 'temp.png')
    img3.save(temp_img_file)
    
    codeImg = cv2.imread(temp_img_file)
    
    # codeImg = cv2.cvtColor(codeImg, cv2.COLOR_BGR2RGB) # may needs to convert from rgb to bgr
    # locate xRange #(1-54)
    xRange = field_x_2_range(center_position[0], convt)
    # split color coded image
    for i in range(yRange):
        ymin = (convt.np_bounds_subplot[xRange][i][2]-yShift) * scaleParam
        ymax = (convt.np_bounds_subplot[xRange][i][3]-yShift) * scaleParam
        
        specifiedIndex = np.where((plyData.data["y"]>ymin) & (plyData.data["y"]<ymax))
        target = plyData.data[specifiedIndex]
        
        if len(target) == 0:
            continue
        
        # get y-axis boundaries in png file
        g_s_i = tInd[specifiedIndex[0]]
        g_start_row = min(g_s_i)/gWid
        g_end_row = max(g_s_i)/gWid
        
        # xRange, yRange to plotNumber
        plotNum = convt.fieldPartition_to_plotNum_32(xRange, i+1)
        
        if min(g_start_row,g_end_row) < 1 or max(g_start_row,g_end_row) > gHei or g_start_row == g_end_row:
            continue
        
        # crop png file
        crop_img = codeImg[g_start_row:g_end_row, :]
        
        out_plot_dir = os.path.join(out_dir, str(plotNum))
        if not os.path.isdir(out_plot_dir):
            os.mkdir(out_plot_dir)
        
        out_file_path = os.path.join(out_plot_dir, 'color_{}.png'.format(str_date))
        cv2.imwrite(out_file_path, crop_img)
        
        # angle hist to chi-p, save chi-fitted plots
        
        angleHist = gen_angle_hist_from_raw(angle_data[i])
        #chi_p = calc_chi_fit_parameter(angleHist)
        beta_p = calc_beta_distribution_value(angleHist)
        if beta_p == None:
            continue
        
        draw_beta_distribution(angleHist, out_dir, str_date, plotNum)
        
        # save chi-p to csv
        saveFile = os.path.join(out_plot_dir, 'saveBeta_{}.txt'.format(plotNum))
        saveHandle = open(saveFile, 'a+')
        saveHandle.write('%s,%f\n'%(str_date, beta_p[2]))
        #saveHandle.write('%s,%f\n'%(str_date, chi_p[1]))
        saveHandle.close()

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

def draw_beta_distribution(angleHist, out_dir, str_date, plotNum):
    

    if np.amax(angleHist) < 5:
        return
    
    if np.isnan(np.min(angleHist)):
        return

    angleSize = 70
    
    angleHist = angleHist / np.sum(angleHist)
    angleHist = angleHist[0:angleSize]
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    x = np.linspace(0.01, 0.99, angleSize)
    y = angleHist
    delta_y = angleSize
    y = y * delta_y
    gmod = Model(beta_distribution)
    try:
        result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
    except ValueError as err:
        print(err.args[0])
    
    rel = np.zeros(4)
    rel[0], rel[1] = beta_mu_nu(tMean, tVar)
    rel[2], rel[3] = beta_mu_nu(result.best_values['tMean'], result.best_values['tVar'])
    
    plt.plot(x,y, 'bo')
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
    plt.title('beta: %f' % (rel[2]))
    textline = 'dotted line: initial fit by beta distribution\nred line: best fitted\nblue plot: LAD from laser scanner'
    plt.annotate(textline, xy=(1, 1), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top')
    out_file = os.path.join(out_dir, str(plotNum), 'fit_{}.png'.format(str_date))
    plt.savefig(out_file)
    plt.close()
    
    return rel


def draw_chi_distribution(angleHist, out_dir, str_date, plotNum):
    
    if np.amax(angleHist) < 5:
        return
    
    if np.isnan(np.min(angleHist)):
        return
    
    angleSize = 70
    angleHist = angleHist / np.sum(angleHist)
    angleHist = angleHist[0:angleSize]
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    init_chi, leaf_lambda = calc_init_chi(tMean)
    x = np.linspace(0.01, 0.99, angleSize)
    y = angleHist
    delta_y = angleSize
    
    y = y * delta_y
    gmod = Model(chi_distribution)
    try:
        result = gmod.fit(y, t=x, init_chi = init_chi, leaf_lambda = leaf_lambda)
    except ValueError as err:
        print(err.args[0])
    
    #print(result.fit_report())
    
    plt.plot(x,y, 'bo')
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
    plt.title('chi-p: %f' % (result.best_values['init_chi']))
    textline = 'dotted line: initial fit by chi-distribution\nred line: best fitted\nblue plot: LAD from laser scanner'
    plt.annotate(textline, xy=(1, 1), xycoords='axes fraction',horizontalalignment='right', verticalalignment='top')
    out_file = os.path.join(out_dir, str(plotNum), 'fit_{}.png'.format(str_date))
    plt.savefig(out_file)
    plt.close()
    
    
    return

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
        tSum += (float(i)/angleHist.size) * angleHist[i]
    tMean = tSum
    
    tVar = 0
    for i in range(angleHist.size):
        tVar += (float(i)/angleHist.size-tMean)*(float(i)/angleHist.size-tMean)*angleHist[i]
        
    tVar = math.sqrt(tVar)
    
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
    
    if np.amax(np_x) < 40:
        return None
    
    if np.isnan(np.min(np_x)):
        return None
    
    angleSize = 70
    angleHist = np_x / np.sum(np_x)
    angleHist = angleHist[0:angleSize]
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    init_chi, leaf_lambda = calc_init_chi(tMean)
    x = np.linspace(0.01, 0.99, angleSize)
    y = angleHist
    delta_y = angleSize

    y = y * delta_y
    gmod = Model(chi_distribution)
    try:
        result = gmod.fit(y, t=x, init_chi = init_chi, leaf_lambda = leaf_lambda)
    except ValueError as err:
        print(err.args[0])
    
    rel = np.zeros(2)
    rel[0] = init_chi
    rel[1] = result.best_values['init_chi']
    
    return rel
    
    
def calc_beta_distribution_value(np_x):
    
    if np.amax(np_x) < 40:
        return None
    
    if np.isnan(np.min(np_x)):
        return None
    
    angleSize = 70
    
    angleHist = np_x / np.sum(np_x)
    angleHist = angleHist[0:angleSize]
    tMean, tVar = get_mean_and_variance_from_angleHist(angleHist)
    x = np.linspace(0.01, 0.99, angleSize)
    y = angleHist
    delta_y = angleSize
    y = y * delta_y
    gmod = Model(beta_distribution)
    try:
        result = gmod.fit(y, t=x, tMean=tMean, tVar=tVar)
    except ValueError as err:
        print(err.args[0])
    
    rel = np.zeros(4)
    rel[0], rel[1] = beta_mu_nu(tMean, tVar)
    rel[2], rel[3] = beta_mu_nu(result.best_values['tMean'], result.best_values['tVar'])
    
    return rel

def estimate_ground_level(plyData, dap=0):
    
    MIN_HEIGHT = 60
    point_z = plyData['z']
    
    if dap < 20:
        z_hist, bounds = np.histogram(point_z, bins=100)
        max_ind = np.argmax(z_hist)
        ground_level = bounds[max_ind-1]
    elif dap > 35:
        ground_level = np.amin(point_z)
    else:
        ground_level = filters.threshold_otsu(point_z, nbins = 100)
    
    ground_level += MIN_HEIGHT
    
    return ground_level

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

def get_area_index(xCoord, yCoord, yOffset, gantry_x, convt):
    
    xPosition = xCoord/1000 + gantry_x
    yPosition = yCoord/1000 + yOffset
    
    count = -1
    
    xRange = 0
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds_subplot[i][0][0]
        xmax = convt.np_bounds_subplot[i][0][1]
        if (xPosition > xmin) and (xPosition <= xmax):
            xRange = i + 1
            break
            
    if xRange == 0:
        return count
        
    for j in range(PLOT_COL_NUM):
        ymin = convt.np_bounds_subplot[xRange-1][j][2]
        ymax = convt.np_bounds_subplot[xRange-1][j][3]
        if (yPosition > ymin) and (yPosition <= ymax):
            count = j + 1
            break
    
    return count

def normals_to_rgb_2(normals):
    
    val = math.atan2(normals[1], normals[0])
    
    h = (val+0.5*math.pi)/(math.pi)#(val+math.pi)/(math.pi*2)
    s = math.acos(normals[2])/(math.pi)
    v = 0.7
    rgb = colorsys.hsv_to_rgb(h, s, v)
    r = int(round(rgb[0]*255))
    g = int(round(rgb[1]*255))
    b = int(round(rgb[2]*255))
    
    return (r,g,b)

def full_day_gen_angle_data(ply_path, json_path, out_dir, str_date, convt, dap):
    
    list_dirs = os.walk(ply_path)
    
    print('ply: {}, json: {}, out: {}'.format(ply_path, json_path, out_dir))
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            p_path = os.path.join(ply_path, d)
            j_path = os.path.join(json_path, d)
            if not os.path.isdir(p_path):
                continue
            
            if not os.path.isdir(j_path):
                continue
            
            create_angle_data_debug(p_path, j_path, out_dir, str_date, convt, dap)
    
    
    return


def leaf_angle_time_series_debug(png_dir, ply_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    d1 = date(2017, 5, 1)
    d2 = date(2017, 9, 1)
    
    deltaDay = d2 - d1
    
    convt = terra_common.CoordinateConverter()
    str_date = str(d1)
    q_flag = convt.bety_query(str_date, True)
        
    for i in range(deltaDay.days+1):
        str_date = str(d1+timedelta(days=i))
        print(str_date)
        #if i < 35:
        #    continue
        ply_path = os.path.join(ply_dir, str_date)
        json_path = os.path.join(png_dir, str_date)
        
        if not os.path.isdir(ply_path):
            continue
        if not os.path.isdir(json_path):
            continue
        
        full_day_gen_angle_data(ply_path, json_path, out_dir, str_date, convt, i)
    
    return
    

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
        
def fail(reason):
    print >> sys.stderr, reason
    
    
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


def find_files(ply_path, json_path):
    json_suffix = os.path.join(json_path, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        print json_path
        fail('Could not find .json file')
        return [], [], [], []
    
    ply_suffix = os.path.join(ply_path, '*-west_0.ply')
    plys = glob(ply_suffix)
    if len(plys) == 0:
        print ply_path
        fail('Could not find west ply file')
        return [], [], [], []
    
    gIm_suffix = os.path.join(json_path, '*-west_0_g.png')
    gIms = glob(gIm_suffix)
    if len(gIms) == 0:
        fail('Could not find -west_0_g.png file')
        return [], [], [], []
    
    pIm_suffix = os.path.join(json_path, '*-west_0_p.png')
    pIms = glob(pIm_suffix)
    if len(pIms) == 0:
        fail('Could not find -west_0_p.png file')
        return [], [], [], []
    
    return jsons[0], plys[0], gIms[0], pIms[0]

def plot_chi_p(in_dir):
    
    list_dirs = os.walk(in_dir)
    
    for d in list_dirs:
        plot_dir = os.path.join(in_dir, d)
        if not os.path.isdir(plot_dir):
            continue
        
        plotNum = int(d)
        save_file = os.path.join(plot_dir, 'saveChi_{}.txt'.format(d))
        file_handle = open(save_file, 'r')
        for line in file_handle:
            fields = line.split(',')
            plotNum = fields[0]
            den = float(fields[1])
            densityList[plotNum].append(den)
        den_handle.close()
    
    
    return

if __name__ == "__main__":
    
    leaf_angle_time_series_debug('/media/zli/Elements/panicle_debug_data/single_scan/raw_data/scanner3DTop', '/media/zli/Elements/panicle_debug_data/single_scan/Level_1/scanner3DTop', '/media/zli/Elements/leaf_angle_debug/Level_2/leaf_angle_beta_var')









