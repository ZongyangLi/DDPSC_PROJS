'''
Created on Oct 1, 2020

@author: zli
'''
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import network
from torch.optim import lr_scheduler
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from datetime import datetime
from image_ops import *
from similarity_ops import *
from scipy import io as sio
from datetime import date
import utils
from dataset import GridMatchDataset
from collections import OrderedDict
batch_size = 1

image_transform = transforms.Compose([transforms.ToTensor()])

class GridMatchDatasetPath(GridMatchDataset):
    def __getitem__(self, index):
        return super().__getitem__(index) + (self.image_paths[index], )
    
    
def init_grid_pool(imgSize, gridSizeVec):
    
    grid_pool = []
    
    for gridSize in gridSizeVec:
        max_index = imgSize // gridSize
        for i in range(max_index):
            for j in range(max_index):
                start_x = i*gridSize
                start_y = j*gridSize
                end_x = (i+1)*gridSize
                end_y = (j+1)*gridSize
                grid_pool.append([start_x,end_x,start_y,end_y])
    
    return grid_pool

def init_grid_pool_largeGrid(imgSize, gridSize, stepSize):
    
    grid_pool = []
    
    max_index = (imgSize - gridSize) // stepSize
    
    for i in range(max_index):
        for j in range(max_index):
            start_x = i*stepSize
            start_y = j*stepSize
            end_x = start_x+gridSize
            end_y = start_y+gridSize
            grid_pool.append([start_x,start_y, end_x,end_y])
    
    return grid_pool

def init_base_rgb_roi_image(rgb_img_path, grid, out_dir):
    
    image = Image.open(rgb_img_path)
    #image = image.resize((224,224))
    im1 = image.crop(grid)
    img_w, img_h = image.size
    background = Image.new('RGB', (img_w, img_h), (0, 0, 0))
    background.paste(im1, (grid[0],grid[1]))
    #background= background.resize((224,224))
    out_path = os.path.join(out_dir, '{}_base.png'.format(grid))
    background.save(out_path)
    return background
    
def embed_single_image(img, whichModel):
    
    data = transforms.ToTensor()(img)
    data = data.unsqueeze(0)
    data = data.cuda()
    fvec = whichModel(data)
    
    return fvec.cpu().detach()

def embed_pair_grid(grid_data, whichModel):
    
    dataloader = torch.utils.data.DataLoader(grid_data, batch_size=batch_size, 
                                         shuffle=False)
    V,L = [],[]
    
    for data, target, image_path in dataloader:
        with torch.no_grad():
            #show_img = transforms.ToPILImage()(data.squeeze(0))
            #show_img.show('show pair thermal')
            data = data.cuda()
            fvec = whichModel(data)
            V.extend(fvec.cpu())
            L.extend([target])
    
    return V,L

def get_scores_for_grids(base_embedding, pair_embeddings, labels, base_grid):
    
    S = []
    
    # noamalize base embedding
    base_embedding = torch.nn.functional.normalize(base_embedding, p=2, dim=1)
    
    for pair_embedding, label in zip(pair_embeddings, labels):
        x1 = label['x1'].item()
        x2 = label['x2'].item()
        y1 = label['y1'].item()
        y2 = label['y2'].item()
        grid = (x1,y1,x2,y2)
        if grid == base_grid:
            print('same box')
        
        # noamalize pair embedding
        pair_embedding = pair_embedding.unsqueeze(0)
        pair_embedding = torch.nn.functional.normalize(pair_embedding, p=2, dim=1)
        
        # dot product
        dot_product = torch.matmul(base_embedding.squeeze(0), pair_embedding.squeeze(0))
        # dist to score
        #s = 1/(1+dist)
        S.append(dot_product.item())
    
    return S

def output_grid_images_by_scores(image_path, scores, labels, out_dir):
    
    np_scores = np.asarray(scores)
    sorted_index = np_scores.argsort()
    
    img = Image.open(image_path)
    image = img.resize((224,224))
    img_w, img_h = image.size
    
    for ind in range(len(sorted_index)):
        label = labels[sorted_index[ind]]
        score = scores[sorted_index[ind]]
        x1 = label['x1'].item()
        x2 = label['x2'].item()
        y1 = label['y1'].item()
        y2 = label['y2'].item()
        grid = (x1,y1,x2,y2)
        crop_img = image.crop(grid)
        background = Image.new('RGB', (img_w, img_h), (0, 0, 0))
        background.paste(crop_img, (grid[0],grid[1]))
        out_path = os.path.join(out_dir, '{}_{}.png'.format(ind, score))
        background.save(out_path)
    
    return

def output_grid_images_by_scores_topN(image_path, scores, labels, out_dir, topNum, base_grid):
    
    np_scores = np.asarray(scores)
    sorted_index = np_scores.argsort()
    
    img = Image.open(image_path)
    image = img
    img_w, img_h = image.size
    
    grid_size = len(sorted_index)
    
    for ind in range(grid_size):
        if ind > topNum:
            break
        label = labels[sorted_index[grid_size-ind-1]]
        score = scores[sorted_index[grid_size-ind-1]]
        x1 = label['x1'].item()
        x2 = label['x2'].item()
        y1 = label['y1'].item()
        y2 = label['y2'].item()
        grid = (x1,y1,x2,y2)
        crop_img = image.crop(grid)
        background = Image.new('RGB', (img_w, img_h), (0, 0, 0))
        background.paste(crop_img, (grid[0],grid[1]))
        out_path = os.path.join(out_dir, '{}_{}_{}.png'.format(base_grid, ind, score))
        background.save(out_path)
    
    return

def generate_score_map(imgSize, scores, labels):
    
    np_map = np.zeros((imgSize, imgSize))
    np_sum = np.ones((imgSize, imgSize))
    
    for score, label in zip(scores, labels):
        x1 = label['x1'].item()
        x2 = label['x2'].item()
        y1 = label['y1'].item()
        y2 = label['y2'].item()
        grid = (x1,y1,x2,y2)
        score = float(score)
        np_map[x1:x2,y1:y2] += score
        np_sum[x1:x2,y1:y2] += 1
        
    np_out = np_map/np_sum
    
    plt.imsave('/home/zli/Desktop/save.png', np_out, cmap='hot')
    
    
    
    
    
    return

def grid_match_jointdata_scan_whole_image(rgb_img_path, thermal_img_path, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    imgSize = 400
    gridSize = 256
    stepSize = 32
    # init grid pool, return: vec of box coordinates
    grid_pool = init_grid_pool_largeGrid(imgSize, gridSize, stepSize)
    
        
    # init image set base on grid pool, both sensor's image
    rgb_grid_data = GridMatchDatasetPath(rgb_img_path, grid_pool, imgSize, image_transform)
    thermal_grid_data = GridMatchDatasetPath(thermal_img_path, grid_pool, imgSize, image_transform)
    
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
    
    output_number = 0
    
    for base_grid in grid_pool:
        # select base roi from RGB
        base_roi_image = init_base_rgb_roi_image(rgb_img_path, base_grid, out_dir)
    
        # get base embedding
        base_embedding = embed_single_image(base_roi_image, model)
    
        # loop thermal data set and get pair embeddings+++
        pair_embeddings, labels = embed_pair_grid(thermal_grid_data, model)
    
        # get a score for the grid
        scores = get_scores_for_grids(base_embedding, pair_embeddings, labels, base_grid)
    
        # output grid images sorted by score
        output_grid_images_by_scores_topN(thermal_img_path, scores, labels, out_dir, output_number, base_grid)
    
    # draw a map base on all the grid score
    #generate_score_map(imgSize, scores, labels)
    
    # output a scored heat map
            
    return
 
def grid_match_jointdata_single_image(rgb_img_path, thermal_img_path, out_dir, model_path):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    imgSize = 224
    gridSize = 128
    stepSize = 8
    # init grid pool, return: vec of box coordinates
    grid_pool = init_grid_pool_largeGrid(imgSize, gridSize, stepSize)
    
        
    # init image set base on grid pool, both sensor's image
    rgb_grid_data = GridMatchDatasetPath(rgb_img_path, grid_pool, imgSize, image_transform)
    thermal_grid_data = GridMatchDatasetPath(thermal_img_path, grid_pool, imgSize, image_transform)
    
    # load model
    model_state_dict = torch.load(model_path)
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    #base_grid = (64,40,128,104)
    base_grid = (64,40,64+128,40+128)
    
    # select base roi from RGB, human selected for now
    base_roi_image = init_base_rgb_roi_image(rgb_img_path, base_grid, out_dir)
    
    # get base embedding
    base_embedding = embed_single_image(base_roi_image, model)
    
    # loop thermal data set and get pair embeddings+++
    pair_embeddings, labels = embed_pair_grid(thermal_grid_data, model)
    
    # get a score for the grid
    scores = get_scores_for_grids(base_embedding, pair_embeddings, labels, base_grid)
    
    # output grid images sorted by score
    output_grid_images_by_scores(thermal_img_path, scores, labels, out_dir)
    
    # draw a map base on all the grid score
    #generate_score_map(imgSize, scores, labels)
    
    # output a scored heat map
            
    return
    
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

from glob import glob
import shutil
def test():
    
    input_folder = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/Joint_statistic_test_sameArg_256_applytransform'
    model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateSameWay30WithFLIP_320_temp3.5/20210106_180416201/epoch_22.pth'
    
    list_dir = os.listdir(input_folder)
    
    
    for d in list_dir:
        sub_dir = os.path.join(input_folder, d)
        if not os.path.isdir(sub_dir):
            continue
        
        print(sub_dir)
        png_suffix = os.path.join(sub_dir, '*.png')
        pngs = glob(png_suffix)
        im1_path = pngs[0]
        im2_path = pngs[1]
        
        out_dir = os.path.join(sub_dir, 'grid_match')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        else:
            shutil.rmtree(out_dir)
        
        grid_match_jointdata_scan_whole_image(im1_path, im2_path, out_dir, model_path)
    
    
    return


def main():
    
    rgb_img_path = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test_0.6m/rgb/34-09-0536-7/2019-06-07.png'
    thermal_img_path = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test_0.6m/thermal/34-09-0536-7/2019-06-07.png'
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/grid_match/RotateSameWay30WithFLIP_320_temp3.5'
    model_path = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateSameWay30WithFLIP_320_temp3.5/20210106_180416201/epoch_22.pth'
    
    grid_match_jointdata_scan_whole_image(rgb_img_path, thermal_img_path, out_dir, model_path)
    
    return

if __name__ == '__main__':
    
    main()
    
    
    