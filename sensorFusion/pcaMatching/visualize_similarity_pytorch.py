'''
Created on Jul 20, 2020

@author: zli
'''
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import math
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import network
from torch.optim import lr_scheduler
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import shutil, random
from datetime import datetime
from image_ops import *
from similarity_ops import *
from scipy import io as sio
from datetime import date
from collections import OrderedDict
import utils
from dataset import SimilarityVisualiztionDataset
batch_size = 1
import torchvision.transforms.functional as TF

from torch.autograd import Variable

imgSize = 320
from scipy.optimize import Bounds
from scipy.optimize import minimize
para_bound_size = imgSize * 0.8
bns = Bounds([-1*para_bound_size, -1*para_bound_size, 0.5], [para_bound_size, para_bound_size, 2])
x0 = [-0.5,-0.5,1.1]

from scipy.optimize import LinearConstraint
cons = [{'type': 'ineq', 'fun': lambda x:  x[0] + 1},
        {'type': 'ineq', 'fun': lambda x:  -x[0] + 1},
        {'type': 'ineq', 'fun': lambda x:  x[1] + 1},
        {'type': 'ineq', 'fun': lambda x:  -x[1] + 1},
        {'type': 'ineq', 'fun': lambda x:  x[2] - 0.5},
        {'type': 'ineq', 'fun': lambda x:  -x[2] + 2}]

#linear_constraint = LinearConstraint([-1*para_bound_size, -1*para_bound_size, 0.5], [para_bound_size, para_bound_size, 2])

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
image_transform = transforms.Compose([transforms.CenterCrop(320),
                                      transforms.ToTensor()])

image_transform_flip = transforms.Compose([MyRotationTransform(angles=[90]),
                                           transforms.CenterCrop(320),
                                           transforms.ToTensor()])

class OPENFusionStereoDatasetPath(SimilarityVisualiztionDataset):
    def __getitem__(self, index):
        return super().__getitem__(index) + (self.image_paths[index], )
    



def visualize_similarity(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    list_dir = os.listdir(in_dir)
    
    for d in list_dir:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        dataset = OPENFusionStereoDatasetPath(in_path, transform=image_transform)
    
        # Run the images through the network, get last conv features
        fvecs, maps, labels, image_path_list = embed(dataset,model)
        
        maps = np.moveaxis(maps,1,3)
        #fvecs = np.array(fvecs)
        #labels = np.array(labels)
    
        # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
        for i in range(len(image_path_list)):
            for j in range(len(image_path_list)):
                conv1 = maps[i]
                conv2 = maps[j]
                conv1 = conv1.reshape(-1,conv1.shape[-1])
                conv2 = conv2.reshape(-1,conv2.shape[-1])
                heatmap1, heatmap2 = compute_spatial_similarity(conv1, conv2)
                
                im1_path = image_path_list[i][0]
                im2_path = image_path_list[j][0]
                
                # Combine the images with the (interpolated) similarity heatmaps.
                im1_with_similarity = combine_image_and_heatmap_noninter(load_and_resize(im1_path),heatmap1)
                im2_with_similarity = combine_image_and_heatmap_noninter(load_and_resize(im2_path),heatmap2)
                
                save_img = combine_horz([im1_with_similarity,im2_with_similarity])
                
                save_img.save(os.path.join(out_path,'b{}-{}_t{}-{}.png'.format(labels[i], i, labels[j], j)))
                #Image.fromarray(im2_with_similarity).save(os.path.join(out_path,'b{}_t{}.png'.format(labels[j],labels[i])))
    
    
    return

def visualize_similarity_smallset(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    list_dir = os.listdir(in_dir)
    
    for d in list_dir:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        dataset = OPENFusionStereoDatasetPath(in_path, transform=image_transform)
    
        # Run the images through the network, get last conv features
        fvecs, maps, labels, image_path_list = embed(dataset,model)
        
        maps = np.moveaxis(maps,1,3)
        #fvecs = np.array(fvecs)
        #labels = np.array(labels)
    
        # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
        for i in range(len(image_path_list)):
            for j in range(len(image_path_list)):
                if abs(i-j) > 2:
                    continue
                if i == j:
                    continue
                conv1 = maps[i]
                conv2 = maps[j]
                conv1 = conv1.reshape(-1,conv1.shape[-1])
                conv2 = conv2.reshape(-1,conv2.shape[-1])
                heatmap1, heatmap2 = compute_spatial_similarity(conv1, conv2)
                
                im1_path = image_path_list[i][0]
                im2_path = image_path_list[j][0]
                
                # Combine the images with the (interpolated) similarity heatmaps.
                im1_with_similarity = combine_image_and_heatmap_noninter(load_and_resize(im1_path),heatmap1)
                im2_with_similarity = combine_image_and_heatmap_noninter(load_and_resize_grayscale(im2_path),heatmap2)
                
                save_img = combine_horz([im1_with_similarity,im2_with_similarity])
                
                save_img.save(os.path.join(out_path,'b{}-{}_t{}-{}.png'.format(os.path.basename(im1_path)[:-4], i, os.path.basename(im2_path)[:-4], j)))
                #Image.fromarray(im2_with_similarity).save(os.path.join(out_path,'b{}_t{}.png'.format(labels[j],labels[i])))
    
    return

def visualize_similarity_jointdata(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    rgb_dir = os.path.join(in_dir, 'rgb')
    thermal_dir = os.path.join(in_dir, 'thermal')
    list_dir = os.listdir(rgb_dir)
    
    for d in list_dir:
        rgb_path = os.path.join(rgb_dir, d)
        thermal_path = os.path.join(thermal_dir, d)
        if not os.path.isdir(rgb_path):
            continue
        if not os.path.isdir(thermal_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        dataset_rgb = OPENFusionStereoDatasetPath(rgb_path, transform=image_transform)
        dataset_thermal = OPENFusionStereoDatasetPath(thermal_path, transform=image_transform)
        
        # Run the images through the network, get last conv features
        fvecs, maps_rgb, labels, image_path_list_rgb = embed(dataset_rgb,model)
        fvecs, maps_thermal, labels, image_path_list_thermal = embed(dataset_thermal,model)
        
        maps_rgb = np.moveaxis(maps_rgb,1,3)
        maps_thermal = np.moveaxis(maps_thermal,1,3)
        #fvecs = np.array(fvecs)
        #labels = np.array(labels)
    
        # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
        for i in range(len(image_path_list_rgb)):
            im1_path = image_path_list_rgb[i][0]
            rgb_time = os.path.basename(im1_path)[:-4]
            rgb_time_int = datetime.strptime(rgb_time, '%Y-%m-%d').date().toordinal()
            for j in range(len(image_path_list_thermal)):
                im2_path = image_path_list_thermal[j][0]
                thermal_time = os.path.basename(im2_path)[:-4]
                thermal_time_int = datetime.strptime(thermal_time, '%Y-%m-%d').date().toordinal()
                if abs(rgb_time_int-thermal_time_int) > 2:
                    continue
                conv1 = maps_rgb[i]
                conv2 = maps_thermal[j]
                conv1 = conv1.reshape(-1,conv1.shape[-1])
                conv2 = conv2.reshape(-1,conv2.shape[-1])
                heatmap1, heatmap2 = compute_spatial_similarity(conv1, conv2)
                
                # Combine the images with the (interpolated) similarity heatmaps.
                im1_with_similarity = combine_image_and_heatmap_noninter(load_and_resize(im1_path),heatmap1)
                im2_with_similarity = combine_image_and_heatmap_noninter(load_and_resize_grayscale(im2_path),heatmap2)
                
                save_img = combine_horz([im1_with_similarity,im2_with_similarity])
                
                save_img.save(os.path.join(out_path,'b{}-{}_t{}-{}.png'.format(os.path.basename(im1_path)[:-4], i, os.path.basename(im2_path)[:-4], j)))
                #Image.fromarray(im2_with_similarity).save(os.path.join(out_path,'b{}_t{}.png'.format(labels[j],labels[i])))
    
    return

def pca_optimization(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    # remove `module.` for parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    
    model.cuda()
    model.eval()
    imsize = (320,320)
    
    rgb_dir = os.path.join(in_dir, 'rgb')
    thermal_dir = os.path.join(in_dir, 'laser')
    list_dir = os.listdir(rgb_dir)
    
    len_dir = len(list_dir)
    for i in range(0,len_dir):
        print(i)
        d = list_dir[i]
        rgb_path = os.path.join(rgb_dir, d)
        thermal_path = os.path.join(thermal_dir, d)
        if not os.path.isdir(rgb_path):
            continue
        if not os.path.isdir(thermal_path):
            continue
            
        list_rgb_files = os.walk(rgb_path)
        for root, dirs, files in list_rgb_files:
            for f in files:
                base_file = os.path.join(rgb_path, f)
                if os.path.exists(base_file):
                    pair_file = os.path.join(thermal_path, f)
                    if os.path.exists(pair_file):
                        out_path = os.path.join(out_dir, d)
                        if not os.path.isdir(out_path):
                            os.mkdir(out_path)
                        image1 = Image.open(base_file)
                        image1 = image1.crop((40,40,360,360))
                        fv1, fm1 = embed_single_image(image1, model)
                        fm1 = np.moveaxis(fm1,1,3)
                        conv1 = fm1.reshape(-1,fm1.shape[-1])
                        
                        image2 = Image.open(pair_file)
                        image2 = image2.crop((40,40,360,360))
                        fv2, fm2 = embed_single_image(image2, model)
                        fm2 = np.moveaxis(fm2,1,3)
                        conv2 = fm2.reshape(-1,fm2.shape[-1])
                        
                        heatmap1, heatmap2 = combine_SVD(conv1,conv2)
                        
                        featPCAIm1 = cv2.resize(heatmap1, (imsize[0], imsize[1]), interpolation=cv2.INTER_LINEAR)
                        
                        
                        featPCAIm2 = cv2.resize(heatmap2, (imsize[0], imsize[1]), interpolation=cv2.INTER_LINEAR)
                        
                        
                        # Combine the images with the (interpolated) similarity heatmaps.
                        im1 = cv2.imread(base_file, 1)
                        im1 = im1[40:360, 40:360]
                        im1 = cv2.resize(im1, (imsize[0], imsize[1]))
                        
                        
                        im2 = cv2.imread(pair_file, 1)
                        im2 = im2[40:360, 40:360]
                        #im2 = cv2.merge((im2,im2,im2))
                        im2 = cv2.resize(im2, (imsize[0], imsize[1]))
                        
                        #featPCAIm1 = featPCAIm1.astype('uint8')
                        #featPCAIm2 = featPCAIm2.astype('uint8')
                        #im1_with_similarity = cv2.addWeighted(im1, 0.5, featPCAIm1, 0.5, 0.0)
                        #im2_with_similarity = cv2.addWeighted(im2, 0.5, featPCAIm2, 0.5, 0.0)
                        #save_img = combine_horz([im1_with_similarity,im2_with_similarity])
                        #save_img.save(os.path.join(out_path,'{}.png'.format(os.path.basename(base_file)[:-4])))
                        
                        save_path = os.path.join(out_path,'{}_fusion.png'.format(os.path.basename(base_file)[:-4]))
                        
                        '''
                        PCA = torch.from_numpy(featPCAIm1)
                        PCAT = torch.from_numpy(featPCAIm2)
                        Im1 = torch.from_numpy(im1)
                        Im2 = torch.from_numpy(im2)
                        pytorch_optimization_frame(PCA, PCAT, Im1, Im2)
                        '''
                        a, b, _ = pca_template_matching(featPCAIm1, featPCAIm2)
                        save_fusion_img(im1, im2, a, b, save_path)
                        '''
                        im_t = cv2.imread(pair_file, 0)
                        im_t = im2[40:360, 40:360]
                        im_t = cv2.merge((im_t,im_t,im_t))
                        im_t = cv2.resize(im_t, (imsize[0], imsize[1]))
                        x0 = [a,b,1.0]
                        res = minimize(scale_translation_optimization_fun, x0, args=(featPCAIm1, featPCAIm2, im1, im_t), method='trust-constr', constraints=cons, options={'gtol': 1e-10, 'disp': True})
                        print(res)
                        '''
                        
                        
    
    return

def save_fusion_img(im1, im2, a, b, save_path):
    
    t_im = np.roll(im2, a, axis=1)
    if a < 0:
        t_im[:,(imgSize+a):] = 0
    else:
        t_im[:, :a] = 0
    t_im = np.roll(t_im, b, axis=0)
    if b < 0:
        t_im[(imgSize+b):,:] = 0
    else:
        t_im[:b, :] = 0
    
    dst = cv2.addWeighted(t_im, 0.2, im1, 0.9, 0.0)
    
    cv2.imwrite(save_path, dst)
    
    return
'''
    i = (a + it) * s
    j = (b + jt) * s
    
    a = x[0], b = x[1], s = x[2]
    
    Im1, opencv rgb image, channel order: b, g, r
    Im2, gray scale thermal image, leaf area lower than ground
'''
def pca_template_matching(pca, pcat):
    
    save_loss = float('inf')
    rel_a = 0
    rel_b = 0
    
    # loop all parameters, a from -96 to 96, b from -16 to 16, s from 1.0 to 1.1
    for s in range(0, 10):  # ignore s for now 
        new_img_size = int((float(s)/100+1.0)*imgSize)
        pcat_ = cv2.resize(pcat, (new_img_size, new_img_size), interpolation=cv2.INTER_LINEAR)
        for a in range(-96, 96, 2):
            for b in range(-32, 32, 2):
                # get shifted pcat value
                t_pca = np.roll(pcat, a, axis=1)
                if a < 0:
                    t_pca[:,(imgSize+a):] = np.NaN
                else:
                    t_pca[:, :a] = np.NaN
                t_pca = np.roll(t_pca, b, axis=0)
                if b < 0:
                    t_pca[(imgSize+b):,:] = np.NaN
                else:
                    t_pca[:b, :] = np.NaN
                
                test_ = t_pca - pca
                loss = np.nansum(test_**2)
                n_count = (imgSize-abs(a)) * (imgSize-abs(b))
                loss = np.sqrt(loss)
                loss /= n_count
        
                if loss < save_loss:
                    save_loss = loss
                    rel_a = a
                    rel_b = b
    
    print('a:{}, b:{}, loss:{}\n'.format(rel_a, rel_b, save_loss))
        
    return rel_a,rel_b, save_loss

def pytorch_optimization_frame(PCA, PCAT, Im1, Im2):
    
    x = torch.rand(1)
    y = torch.rand(1)
    z = torch.rand(1)
    x= Variable(x, requires_grad=True)
    y= Variable(y, requires_grad=True)
    z= Variable(z, requires_grad=True)
    
    lr = 0.001
    for i in range(20000):
        loss = pytorch_scale_translation_loss(x,y,z, PCA, PCAT, Im1, Im2)
        loss.backward()
        
        x.data = x.data - lr*x.grad.data
        y.data = y.data - lr*y.grad.data
        z.data = z.data - lr*z.grad.data
        
        x.grad.data.zero_()
        y.grad.data.zero_()
        z.grad.data.zero_()
        
    x,y,z
    
    return

def pytorch_scale_translation_loss(x,y,z, PCA, PCAT, Im1, Im2):
    
    alpha = 1
    beta = 1
    delta = 1
    
    loss_pca = 0
    loss_thermal = 0
    n_count = 0
    LEAF_THRESHOLD = 3
    
    leaf_n_count = 0
    total_leaf_temp_val = 0
    ground_n_count = 0
    total_ground_temp_val = 0
    
    max_pca = 0
    
    print('x: {} y: {} z: {}'.format(x,y,z))
    
    # get shifted coordinate from rgb to thermal , (i,j) rgb coordinate, (i_t, j_t) thermal coordinate
    for i in range(0, imgSize):
        for j in range(0, imgSize):
            i_t = (x*para_bound_size+z*i).long()
            j_t = (y*para_bound_size+z*j).long()
            if i_t > 0 and i_t < imgSize and j_t > 0 and j_t < imgSize:
                n_count += 1
                pca_t = PCAT[i_t, j_t].double()
                pca_r = PCA[i,j].double()
                # sum loss of pca
                pca_dist = torch.sum((pca_t - pca_r)**2)/3
                loss_pca += pca_dist
                if pca_dist > max_pca:
                    max_pca = pca_dist
    '''
                # sum thermal(x,y) for leaf or ground
                whether_leaf_value = int(Im1[i,j][1])-Im1[i,j][2]
                if whether_leaf_value > LEAF_THRESHOLD:
                    leaf_n_count += 1
                    total_leaf_temp_val += Im2[i_t, j_t][0]
                if whether_leaf_value < LEAF_THRESHOLD * (-1):
                    ground_n_count += 1
                    total_ground_temp_val += Im2[i_t, j_t][0]
    '''
    # Loss 1, loss from pca
    loss_1 = loss_pca / (n_count * max_pca)
    
    # Loss 2, loss from im
    #ave_leaf_ground_dist = total_leaf_temp_val/float(leaf_n_count) - total_ground_temp_val/float(ground_n_count)
    #loss_2 = math.pow(0.5, ave_leaf_ground_dist)
    
    # OV penalty
    ov_loss = max(0, 0.5-n_count/(imgSize**2))
    
    total_loss = alpha*loss_1 + delta*ov_loss
    
    print(total_loss)
    
    return total_loss
'''
    i = a + s * it
    j = b + s * jt
    
    a = x[0], b = x[1], s = x[2]
    
    Im1, opencv rgb image, channel order: b, g, r
    Im2, gray scale thermal image, leaf area lower than ground
'''
def scale_translation_optimization_fun(x, PCA, PCAT, Im1, Im2):
    
    alpha = 1
    beta = 1
    delta = 1
    
    loss_pca = 0
    loss_thermal = 0
    n_count = 0
    LEAF_THRESHOLD = 3
    
    leaf_n_count = 0
    total_leaf_temp_val = 0
    ground_n_count = 0
    total_ground_temp_val = 0
    
    max_pca = 0
    
    print('x: {}'.format(x))
    
    # get shifted coordinate from rgb to thermal , (i,j) rgb coordinate, (i_t, j_t) thermal coordinate
    for i in range(0, imgSize):
        for j in range(0, imgSize):
            i_t = int(x[0]+x[2]*i)
            j_t = int(x[1]+x[2]*j)
            if i_t > 0 and i_t < imgSize and j_t > 0 and j_t < imgSize:
                n_count += 1
                pca_t = PCAT[i_t, j_t]
                pca_r = PCA[i,j]
                # sum loss of pca
                pca_dist = np.sum((pca_t - pca_r)**2)
                loss_pca += pca_dist 
    
                # sum thermal(x,y) for leaf or ground
                whether_leaf_value = int(Im1[i,j][1])-Im1[i,j][2]
                if whether_leaf_value > LEAF_THRESHOLD:
                    leaf_n_count += 1
                    total_leaf_temp_val += Im2[i_t, j_t][0]
                if whether_leaf_value < LEAF_THRESHOLD * (-1):
                    ground_n_count += 1
                    total_ground_temp_val += Im2[i_t, j_t][0]
    
    # Loss 1, loss from pca
    loss_1 = np.sqrt(loss_pca) / n_count
    
    # Loss 2, loss from im
    ave_leaf_ground_dist = total_leaf_temp_val/float(leaf_n_count) - total_ground_temp_val/float(ground_n_count)
    loss_2 = math.pow(0.5, ave_leaf_ground_dist)
    
    # OV penalty
    ov_loss = max(0, 0.5-n_count/(imgSize**2))
    
    total_loss = alpha*loss_1 + beta*loss_2 + delta*ov_loss
    
    print(total_loss)
    
    return total_loss



def visualize_pca_jointdata(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    # remove `module.` for parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    rgb_dir = os.path.join(in_dir, 'rgb')
    thermal_dir = os.path.join(in_dir, 'laser')
    list_dir = os.listdir(rgb_dir)
    
    len_dir = len(list_dir)
    
    for i in range(0,len_dir):
        print(i)
        d = list_dir[i]
        print(d)
        rgb_path = os.path.join(rgb_dir, d)
        thermal_path = os.path.join(thermal_dir, d)
        if not os.path.isdir(rgb_path):
            continue
        if not os.path.isdir(thermal_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        dataset_rgb = OPENFusionStereoDatasetPath(rgb_path, transform=image_transform)
        dataset_thermal = OPENFusionStereoDatasetPath(thermal_path, transform=image_transform)
        
        # Run the images through the network, get last conv features
        fvecs, maps_rgb, labels, image_path_list_rgb = embed(dataset_rgb,model)
        fvecs, maps_thermal, labels, image_path_list_thermal = embed(dataset_thermal,model)
        
        maps_rgb = np.moveaxis(maps_rgb,1,3)
        maps_thermal = np.moveaxis(maps_thermal,1,3)
        #fvecs = np.array(fvecs)
        #labels = np.array(labels)
        
        imsize = (320,320)
    
        # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
        for i in range(len(image_path_list_rgb)):
            im1_path = image_path_list_rgb[i][0]
            rgb_time = os.path.basename(im1_path)[:-4]
            rgb_time_int = datetime.strptime(rgb_time, '%Y-%m-%d').date().toordinal()
            for j in range(len(image_path_list_thermal)):
                im2_path = image_path_list_thermal[j][0]
                thermal_time = os.path.basename(im2_path)[:-4]
                thermal_time_int = datetime.strptime(thermal_time, '%Y-%m-%d').date().toordinal()
                if abs(rgb_time_int-thermal_time_int) > 2:
                    continue
                conv1 = maps_rgb[i]
                conv2 = maps_thermal[j]
                heatmap1, heatmap2 = combine_SVD(conv1,conv2)
                
                featPCAIm1 = cv2.resize(heatmap1, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
                featPCAIm1 = featPCAIm1.astype('uint8')
                
                featPCAIm2 = cv2.resize(heatmap2, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
                featPCAIm2 = featPCAIm2.astype('uint8')
                
                # Combine the images with the (interpolated) similarity heatmaps.
                im1 = cv2.imread(im1_path, 1)
                #im1 = cv2.flip(im1, 0)
                #im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
                im1 = im1[40:360, 40:360]
                im1 = cv2.resize(im1, (imsize[0], imsize[1]))
                im1_with_similarity = cv2.addWeighted(im1, 0.5, featPCAIm1, 0.5, 0.0)
                
                im2 = cv2.imread(im2_path, 0)
                im2 = im2[40:360, 40:360]
                im2 = cv2.merge((im2,im2,im2))
                im2 = cv2.resize(im2, (imsize[0], imsize[1]))
                im2_with_similarity = cv2.addWeighted(im2, 0.5, featPCAIm2, 0.5, 0.0)
                
                save_img = combine_horz([im1_with_similarity,im2_with_similarity])
                
                save_img.save(os.path.join(out_path,'b{}-{}_t{}-{}.png'.format(os.path.basename(im1_path)[:-4], i, os.path.basename(im2_path)[:-4], j)))
                #Image.fromarray(im2_with_similarity).save(os.path.join(out_path,'b{}_t{}.png'.format(labels[j],labels[i])))
    
    return

def pca_vis_by_score(out_dir, model_path):
    
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    '''
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    '''
    # remove `module.` for parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    task_name = 'RotateDiffWay30WithFLIP_320_temp3.5'
    load_result_ep = 21
    result_dict = torch.load('/home/zli/WorkSpace/PyWork/pytorch-py3/reverse-pheno-master/results/{}_ep_{}.pth'.format(task_name, load_result_ep))
    vectors = result_dict['vectors']
    labels = (result_dict['labels'])
    image_paths = result_dict['image_paths']
    
    total_nums = len(vectors)
    imsize = (320,320)
    
    for i in range(15,total_nums):
        
        print(i)
        
        base_vec = vectors[i]
        base_label = labels['plot'][i]
        base_sensor = labels['sensor'][i]
        dist_rec = np.zeros(total_nums)
        for j in range(total_nums):
            pair_sensor = labels['sensor'][j]
            loop_vec = vectors[j]
            if base_sensor is pair_sensor:
                dist_rec[j] = float('inf')
            else:
                dist_rec[j] = np.linalg.norm(loop_vec-base_vec)   
                # dot product
                #dist_rec[j] = abs(np.matmul(loop_vec, base_vec))
            
        min_index = dist_rec.argsort()[:4]
        mid_ind = int(total_nums/2)
        start_ind = mid_ind - 5
        max_index = dist_rec.argsort()[start_ind:mid_ind]
        
        out_path = os.path.join(out_dir, str(i))
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        
        src_path_base = image_paths[i]
        
        image1 = Image.open(src_path_base)
        image1 = image1.crop((40,40,360,360))
        fv1, fm1 = embed_single_image(image1, model)
        fm1 = np.moveaxis(fm1,1,3)
        conv1 = fm1.reshape(-1,fm1.shape[-1])
        
        for ind in range(len(min_index)):
            val = min_index[ind]
            src_path_pair = image_paths[val]
            if image_paths[val] == image_paths[i]:
                continue
        
            image2 = Image.open(src_path_pair)
            image2 = image2.crop((40,40,360,360))
            #image1.show()
            fv2, fm2 = embed_single_image(image2, model)
            fm2 = np.moveaxis(fm2,1,3)
            conv2 = fm2.reshape(-1,fm2.shape[-1])
            heatmap1, heatmap2 = combine_SVD(conv1,conv2)
            
            featPCAIm1 = cv2.resize(heatmap1, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
            featPCAIm1 = featPCAIm1.astype('uint8')
            
            featPCAIm2 = cv2.resize(heatmap2, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
            featPCAIm2 = featPCAIm2.astype('uint8')
            
            # Combine the images with the (interpolated) similarity heatmaps.
            im1 = cv2.imread(src_path_base, 1)
            im1 = im1[40:360, 40:360]
            im1_with_similarity = cv2.addWeighted(im1, 0.5, featPCAIm1, 0.5, 0.0)
            
            im2 = cv2.imread(src_path_pair, 0)
            im2 = im2[40:360, 40:360]
            im2 = cv2.merge((im2,im2,im2))
            im2_with_similarity = cv2.addWeighted(im2, 0.5, featPCAIm2, 0.5, 0.0)
            
            save_img = combine_horz([im1_with_similarity,im2_with_similarity])
            save_path = os.path.join(out_path, 'close_{}.png'.format(ind))
            save_img.save(save_path)
        
        
        for c in range(len(max_index)):
            val = max_index[c]
            src_path_pair = image_paths[val]
            if image_paths[val] == image_paths[i]:
                continue
        
            image2 = Image.open(src_path_pair)
            image2 = image2.crop((40,40,360,360))
            #image1.show()
            fv2, fm2 = embed_single_image(image2, model)
            fm2 = np.moveaxis(fm2,1,3)
            conv2 = fm2.reshape(-1,fm2.shape[-1])
            heatmap1, heatmap2 = combine_SVD(conv1,conv2)
            
            featPCAIm1 = cv2.resize(heatmap1, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
            featPCAIm1 = featPCAIm1.astype('uint8')
            
            featPCAIm2 = cv2.resize(heatmap2, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
            featPCAIm2 = featPCAIm2.astype('uint8')
            
            # Combine the images with the (interpolated) similarity heatmaps.
            im1 = cv2.imread(src_path_base, 1)
            im1 = im1[40:360, 40:360]
            im1_with_similarity = cv2.addWeighted(im1, 0.7, featPCAIm1, 0.5, 0.0)
            
            im2 = cv2.imread(src_path_pair, 0)
            im2 = im2[40:360, 40:360]
            im2 = cv2.merge((im2,im2,im2))
            im2_with_similarity = cv2.addWeighted(im2, 0.7, featPCAIm2, 0.5, 0.0)
            
            save_img = combine_horz([im1_with_similarity,im2_with_similarity])
            save_path = os.path.join(out_path, 'far_{}.png'.format(c))
            save_img.save(save_path)
            

    return


from glob import glob

def pca_vis_imgList(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    filelist = getFileNamesRecursive(in_dir)
    
    #nF = len(filelist)
    
    # load model
    model = loadModelParallel(model_path)
    
    # create convList
    convList = create_convList(filelist, model)
    
    # compute SVD
    heatmapList = SVD_whole(convList)
    
    # output combined image
    save_combined_image(filelist, heatmapList, out_dir)
    
    return

def save_combined_image(filelist, heatmapList, out_dir):
    
    imsize = (320,320)
    ind = 0
    
    for file_path, heatmap in zip(filelist, heatmapList):
        featPCAIm = cv2.resize(heatmap, (imsize[0], imsize[1]), interpolation=cv2.INTER_NEAREST)
        featPCAIm = featPCAIm.astype('uint8')
        
        # Combine the images with the (interpolated) similarity heatmaps.
        im1 = cv2.imread(file_path, 1)
        im1 = im1[40:360, 40:360]
        im1_with_similarity = cv2.addWeighted(im1, 0.7, featPCAIm, 0.5, 0.0)
        
        #save_img = combine_horz([im1_with_similarity,im2_with_similarity])
        save_path = os.path.join(out_dir, '{}.png'.format(ind))
        ind += 1
        cv2.imwrite(save_path, im1_with_similarity)
        #im1_with_similarity.save(save_path)
    
    return

def loadModelParallel(model_path):
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    # remove `module.` for parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    return model

def create_convList(fileList, model):
    
    convList = []
    
    for img_path in fileList:
        image = Image.open(img_path)
        image = image.crop((40,40,360,360))
        fv, fm = embed_single_image(image, model)
        fm = np.moveaxis(fm,1,3)
        conv = fm.reshape(-1,fm.shape[-1])
        convList.append(conv)
        
    return convList

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
        
        '''
        png_suffix = os.path.join(thermal_dir, '*.png')
        pngs = glob(png_suffix)
        for png in pngs:
            filelist.append(png)
        '''
    
    return filelist

def visualize_sim_pairs_in_folder(input_folder, model_path):
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    list_dir = os.listdir(input_folder)
    
    for d in list_dir:
        sub_dir = os.path.join(input_folder, d)
        if not os.path.isdir(sub_dir):
            continue
        png_suffix = os.path.join(sub_dir, '*.png')
        pngs = glob(png_suffix)
        im1_path = pngs[0]
        im2_path = pngs[1]
        
        image1 = Image.open(im1_path)
        image2 = Image.open(im2_path)
        image1 = image1.crop((40,40,360,360))
        image2 = image2.crop((40,40,360,360))
        
        #image1.show()
        
        fv1, fm1 = embed_single_image(image1, model)
        fv2, fm2 = embed_single_image(image2, model)
        
        fm1 = np.moveaxis(fm1,1,3)
        fm2 = np.moveaxis(fm2,1,3)
        
        conv1 = fm1.reshape(-1,fm1.shape[-1])
        conv2 = fm2.reshape(-1,fm2.shape[-1])
        #heatmap1, heatmap2 = compute_spatial_similarity(conv1, conv2)
        heatmap1, heatmap2 = combine_SVD(conv1,conv2)
        
        # Combine the images with the (interpolated) similarity heatmaps.
        im1_with_similarity = combine_image_and_heatmap_noninter(load_and_resize_grayscale(im1_path),heatmap1)
        im2_with_similarity = combine_image_and_heatmap_noninter(load_and_resize_grayscale(im2_path),heatmap2)
        
        save_img = combine_horz([im1_with_similarity,im2_with_similarity])
        
        save_img.save(os.path.join(sub_dir,'{}_simVis.png'.format(os.path.basename(im1_path)[:-4])))
    
    return

def output_image_and_activations_jointdata(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    rgb_dir = os.path.join(in_dir, 'rgb')
    thermal_dir = os.path.join(in_dir, 'thermal')
    list_dir = os.listdir(rgb_dir)
    
    for d in list_dir:
        rgb_path = os.path.join(rgb_dir, d)
        thermal_path = os.path.join(thermal_dir, d)
        if not os.path.isdir(rgb_path):
            continue
        if not os.path.isdir(thermal_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        
        out_path_rgb = os.path.join(out_dir, d, 'rgb')
        if not os.path.isdir(out_path_rgb):
            os.mkdir(out_path_rgb)
            
        out_path_thermal = os.path.join(out_dir, d, 'thermal')
        if not os.path.isdir(out_path_thermal):
            os.mkdir(out_path_thermal)
    
        dataset_rgb = OPENFusionStereoDatasetPath(rgb_path, transform=image_transform)
        dataset_thermal = OPENFusionStereoDatasetPath(thermal_path, transform=image_transform)
        
        # Run the images through the network, get last conv features
        fvecs, maps_rgb, labels, image_path_list_rgb = embed(dataset_rgb,model)
        fvecs, maps_thermal, labels, image_path_list_thermal = embed(dataset_thermal,model)
        
        maps_rgb = np.moveaxis(maps_rgb,1,3)
        maps_thermal = np.moveaxis(maps_thermal,1,3)
        
        for i in range(len(image_path_list_rgb)):
            conv1 = maps_rgb[i]
            im1_path = image_path_list_rgb[i][0]
            basename = os.path.basename(im1_path)[:-4]
            conv_file_path = os.path.join(out_path_rgb, '{}.pkl'.format(basename))
            mat_file_path = os.path.join(out_path_rgb, '{}.mat'.format(basename))
            img_path = os.path.join(out_path_rgb, '{}.png'.format(basename))
            
            sio.savemat(mat_file_path, {"conv": conv1})
            conv_df = pd.DataFrame({"conv": [conv1]})
            shutil.copyfile(im1_path, img_path)
            conv_df.to_pickle(conv_file_path)
            
        for i in range(len(image_path_list_thermal)):
            conv1 = maps_thermal[i]
            im1_path = image_path_list_thermal[i][0]
            basename = os.path.basename(im1_path)[:-4]
            conv_file_path = os.path.join(out_path_thermal, '{}.pkl'.format(basename))
            mat_file_path = os.path.join(out_path_thermal, '{}.mat'.format(basename))
            img_path = os.path.join(out_path_thermal, '{}.png'.format(basename))
            
            sio.savemat(mat_file_path, {"conv": conv1})
            conv_df = pd.DataFrame({"conv": [conv1]})
            shutil.copyfile(im1_path, img_path)
            conv_df.to_pickle(conv_file_path)
            
    return

def output_image_and_activations(in_dir, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    
    list_dir = os.listdir(in_dir)
    
    for d in list_dir:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        out_path = os.path.join(out_dir, d)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        dataset = OPENFusionStereoDatasetPath(in_path, transform=image_transform)
    
        # Run the images through the network, get last conv features
        fvecs, maps, labels, image_path_list = embed(dataset,model)
        
        maps = np.moveaxis(maps,1,3)
        
        for i in range(len(image_path_list)):
            conv1 = maps[i]
            im1_path = image_path_list[i][0]
            basename = os.path.basename(im1_path)[:-4]
            conv_file_path = os.path.join(out_path, '{}.pkl'.format(basename))
            mat_file_path = os.path.join(out_path, '{}.mat'.format(basename))
            img_path = os.path.join(out_path, '{}.png'.format(basename))
            
            sio.savemat(mat_file_path, {"conv": conv1})
            conv_df = pd.DataFrame({"conv": [conv1]})
            shutil.copyfile(im1_path, img_path)
            conv_df.to_pickle(conv_file_path)
            
    return

def embed_single_image(img, whichModel):
    
    data = transforms.ToTensor()(img)
    data = data.unsqueeze(0)
    data = data.cuda()
    fvec = whichModel(data)
    
    modules = list(whichModel.children())[:-2]
    last_conv = nn.Sequential(*modules)
    last_conv = last_conv.cuda()
    fmap = last_conv(data)
    
    return fvec.cpu().detach(), fmap.cpu().detach().numpy()

def embed(dsets,whichModel):

    dataloader = torch.utils.data.DataLoader(dsets, batch_size=batch_size, 
                                             shuffle=False)
    
    # fc output
    fc = whichModel.fc.state_dict()
    # last conv output
    modules = list(whichModel.children())[:-2]
    last_conv = nn.Sequential(*modules)
    last_conv = last_conv.cuda()
    # iterate batch
    V,M,L = [],[],[]
    image_path_list = []
    
    
    for data, target, image_path in dataloader:
        with torch.no_grad():
            data = data.cuda()
            fvec = whichModel(data)
            fmap = last_conv(data)
            
            V.extend(fvec.cpu().numpy())
            M.extend(fmap.cpu().numpy())
            L.extend(target['plot'])
            image_path_list.append(image_path)

    return V, M, L, image_path_list


def test():
    
    input_folder = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/Joint_statistic_test_sameArg_256_applytransform'
    model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss/20201102_201851106/epoch_26.pth'
    
    visualize_sim_pairs_in_folder(input_folder, model_path)
    
    return

def main():
    
    #in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test_double'
    #out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/pca_vis/rgb_thermal_final_pca'
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_rgb_laser/test'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/pca_vis/laser_rgb_vertical_pca_vis_0408/'
    #model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateDiffWay33_temp3.5_linear/20210122_192953695/epoch_33.pth'
    model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/join_laser_rgb/verticalView_20210408/epoch_13.pth'
    #output_image_and_activations_jointdata(in_dir, out_dir, model_path)
    visualize_pca_jointdata(in_dir, out_dir, model_path) 
    #pca_optimization(in_dir, out_dir, model_path)
    #pca_vis_by_score(out_dir, model_path)
    
    return

def main_2():
    
    in_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/imgs_with_lastConv_feature_few'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/pca_vis/multi_features_vis'
    model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateSameWay30WithFLIP_320_temp3.5/20210106_180416201/epoch_22.pth'
    
    pca_vis_imgList(in_dir, out_dir, model_path)
    
    return

if __name__ == '__main__':
    
    main()
    
    
    
    