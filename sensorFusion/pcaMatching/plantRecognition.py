'''
Created on Nov 12, 2020

@author: zli
'''
import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from scipy.sparse.linalg import svds

def test():
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/imgs_with_lastConv_feature_few'
    
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/pca_vis_multiPC'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_feature_magnitude = os.path.join(out_dir, 'featMagnitude')
    if not os.path.isdir(out_feature_magnitude):
        os.mkdir(out_feature_magnitude)
        
    out_PCAIm = os.path.join(out_dir, 'featPCAIm[0:3]')
    if not os.path.isdir(out_PCAIm):
        os.mkdir(out_PCAIm)
    
    filelist = getFileNamesRecursive(in_dir)
    
    nF = len(filelist)
    CC = np.zeros((nF, 10, 10, 2048))
    
    # load data
    for i in range(nF):
        imfile = filelist[i]
        convfile = imfile[:-3] + 'pkl'
        if not os.path.isfile(convfile):
            continue
        
        im = cv2.imread(imfile, 1)
        imsize = im.shape
        conv_pkl = pd.read_pickle(convfile)
        conv_data = conv_pkl.conv.values[0]
        CC[i,:,:,:] = conv_data
        '''
        # feature magnitude
        featMags = np.sqrt(np.sum(np.power(conv_data, 2), 2))
        
        featMagsScaled = featMags/featMags.max()
        
        featMagIm = cv2.resize(featMagsScaled, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
        
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)/255
        hsv[:,:,1] = 0.55
        hsv[:,:,0] = featMagIm * 0.7
        hsv = hsv * 255
        hsv = hsv.astype('uint8')
        im2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        save_path = os.path.join(out_feature_magnitude, '{}.png'.format(i))
        cv2.imwrite(save_path, im2)
        '''
        
    # PCA
    CC2 = np.reshape(CC, (-1,2048))
    row_mean = np.mean(CC2, axis=0)
    CC3 = CC2 - row_mean
    
    u, s, v = svds(CC3, k=30)
    s = np.diag(s)
    
    uxs = np.matmul(u,s)
    PCA_F = np.reshape(np.matmul(u,s), (-1, 10, 10, 30))
    PCA_vis = 0.5+0.5*PCA_F/np.max(np.abs(PCA_F))
    
    for i in range(nF):
        imfile = filelist[i]
        im = cv2.imread(imfile, 1)
        imsize = im.shape
        feats_1 = PCA_F[i, :, :, 0]
        zero_feat = np.zeros((10,10))
        feats = cv2.merge((feats_1,zero_feat,zero_feat))
        featMagsScaled = 0.5+0.5*feats/np.max(np.abs(feats))
        featPCAIm = cv2.resize(featMagsScaled, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
        featPCAIm = featPCAIm * 255
        featPCAIm = featPCAIm.astype('uint8')
        
        combineImg = cv2.addWeighted(im, 0.5, featPCAIm, 0.5, 0.0)
        save_path = os.path.join(out_PCAIm, '{}.png'.format(i))
        cv2.imwrite(save_path, combineImg)
        
    
    return

def getFileNamesRecursive(in_dir):
    
    list_dirs = os.listdir(in_dir)
    
    filelist = []
    
    for d in list_dirs:
        rgb_dir = os.path.join(in_dir, d, 'rgb')
        thermal_dir = os.path.join(in_dir, d, 'thermal')
        
        if not os.path.isdir(rgb_dir) or not os.path.isdir(thermal_dir):
            continue
        
        png_suffix = os.path.join(rgb_dir, '*.png')
        pngs = glob(png_suffix)
        for png in pngs:
            filelist.append(png)
            
        png_suffix = os.path.join(thermal_dir, '*.png')
        pngs = glob(png_suffix)
        for png in pngs:
            filelist.append(png)
    
    
    return filelist


if __name__ == '__main__':
    
    test()