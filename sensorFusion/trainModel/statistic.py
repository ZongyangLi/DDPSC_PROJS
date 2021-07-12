'''
Created on Jul 9, 2020

@author: zli
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import network
from torch.optim import lr_scheduler
from torch import optim
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from loss import MyNCALoss

from datetime import date
from collections import OrderedDict
import utils
from dataset import OPENFusionJointDataset_test, OPENFusionStereoDataset
batch_size = 20
#image_transform = transforms.ToTensor()
class OPENFusionStereoDatasetPath(OPENFusionJointDataset_test):
    def __getitem__(self, index):
        return super().__getitem__(index) + (self.image_paths[index], )
# dataset = OPENScanner3dDatasetPath('./datasets/OPEN', start_date=start_date, end_date=end_date, exclude_date=exclude_date,
#                                                       transform=image_transform, exclude_cultivar=exclude_cultivar)

image_transform = transforms.Compose([transforms.CenterCrop(320),
                                      transforms.ToTensor()])


def get_output():
    
    dataset = OPENFusionStereoDatasetPath('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test_double', transform=image_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=4)
    
    #model_state_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateSameWay30WithFLIP_320_temp3.5/20210106_180416201/epoch_22.pth')
    #model_state_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateDifferentWay30WithFLIP_320_temp3.5/20210110_161216957/epoch_21.pth')
    model_state_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateDiffWay33_temp3.5_linear/20210122_192953695/epoch_33.pth')
    model = network.resnet_50_embedding()
    
    # remove `module.` for parallel
    new_state_dict = OrderedDict()
    for k, v in model_state_dict['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(model_state_dict['model_state_dict'])
    
    # inference
    #model.load_state_dict(model_state_dict['model_state_dict'])
    model.cuda()
    model.eval()
    vector_list = []
    label_list = []
    image_path_list = []
    with torch.no_grad():
        for data, target, image_path in tqdm(dataloader, total=len(dataloader)):
            data = data.cuda()
            vector_list.append(model(data))
            label_list.append(target)
            image_path_list.append(image_path)
    #vectors = torch.cat(vector_list, 0)
    labels = {}
    for l in label_list:
        for key, val in l.items():
            if key not in labels:
                labels[key] = []
            labels[key].extend(val)
    image_paths = np.concatenate(image_path_list)
    vectors_norm = [torch.nn.functional.normalize(d, p=2, dim=1) for d in vector_list]
    vectors = torch.cat(vectors_norm, 0).cpu().numpy()
    
    

    torch.save({'vectors': vectors,
            'labels': labels,
            'image_paths': image_paths}, './results/{}_ep_{}.pth'.format('RotateDiffWay33_temp3.5_linear', 33))
    
    return

def loss_plot():
    
    dataset = OPENFusionJointDataset('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test', transform=image_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=1)
    
    model_state_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss/20200915_221501874/epoch_11.pth')
    model = network.resnet_50_embedding()
    model.load_state_dict(model_state_dict['model_state_dict'])
    
    loss_func = MyNCALoss()
    
    # inference
    #model.load_state_dict(model_state_dict['model_state_dict'])
    model.cuda()
    model.eval()
    vector_list = []
    label_list = []
    loss_list = []
    with torch.no_grad():
        for data, target in tqdm(dataloader, total=len(dataloader)):
            data = data.cuda()
            model_output = model(data)
            loss = loss_func(model_output, target)
            loss = loss.item()
            vector_list.append(model_output)
            loss_list.append(loss)
            label_list.append(target)
            
    vectors = torch.cat(vector_list, 0)
    
    labels = {}
    for l in label_list:
        for key, val in l.items():
            if key not in labels:
                labels[key] = []
            labels[key].extend(val)
    
    return

import shutil

SAVE_FLAG = False

def compute_dist_matrix():
    
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/Joint_statistic_RotateDiffWay33_temp3.5_linear'
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    task_name = 'RotateDiffWay33_temp3.5_linear'
    load_result_ep = 33
    result_dict = torch.load('./results/{}_ep_{}.pth'.format(task_name, load_result_ep))
    vectors = result_dict['vectors']
    labels = (result_dict['labels'])
    image_paths = result_dict['image_paths']
    
    total_nums = len(vectors)
    save_num = 1
    total_correct = 0
    
    for i in range(total_nums):
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
                #dist_rec[j] = np.linalg.norm(loop_vec-base_vec)   
                # dot product
                dist_rec[j] = (1-np.matmul(loop_vec, base_vec))
            
        min_index = dist_rec.argsort()[:1]
        out_path = os.path.join(out_dir, str(i))
        src_path_base = image_paths[i]
        dst_path_base = os.path.join(out_path, '{}.png'.format(base_label))
        
        check_flag = False
        for ind in range(len(min_index)):
            val = min_index[ind]
            src_path_pair = image_paths[val]
            if image_paths[val] == image_paths[i]:
                continue
            
            pair_label = labels['plot'][val]
            dst_file_name = '{}_{}_{}.png'.format(pair_label, ind, dist_rec[val])
            dst_path_pair = os.path.join(out_path, dst_file_name)
            
            if not check_flag:
                if (base_label == pair_label):
                    total_correct += 1
                    check_flag = True
                    if SAVE_FLAG:
                        if not os.path.isdir(out_path):
                            os.mkdir(out_path)
                        shutil.copyfile(src_path_base, dst_path_base)
                        shutil.copyfile(src_path_pair, dst_path_pair)
                
    print('total correct: {}'.format(total_correct))
    print('correct ratio: {}'.format(float(total_correct)/total_nums))

    return

from numpy import linspace

def build_nearest_neighbor_hist():
    
    out_dir = '/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/sensor_fusion_outputs/Joint_statistic_test'
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    task_name = 'RotateOtherWay90'
    load_result_ep = 29
    result_dict = torch.load('./results/{}_ep_{}.pth'.format(task_name, load_result_ep))
    vectors = result_dict['vectors']
    labels = (result_dict['labels'])
    image_paths = result_dict['image_paths']
    
    total_nums = len(vectors)
    
    hist_100 = np.zeros(100)
    
    for i in range(total_nums):
        base_vec = vectors[i]
        base_label = labels['plot'][i]
        #print('base_label: {}'.format(base_label))
        base_sensor = labels['sensor'][i]
        dist_rec = np.zeros(total_nums)
        for j in range(total_nums):
            pair_sensor = labels['sensor'][j]
            loop_vec = vectors[j]
            if base_sensor is pair_sensor:
                dist_rec[j] = float('inf')
            else:
                dist_rec[j] = np.linalg.norm(loop_vec-base_vec)
            
        min_index = dist_rec.argsort()
        for z in range(100):
            val = min_index[z]
            pair_label = labels['plot'][val]
            if (base_label == pair_label):
                hist_100[z] = hist_100[z] + 1
                print(z)
                break
            
    print(hist_100)
    x = linspace(0, 99, 100)
    y = hist_100
    plt.plot(x,y)
    return
        

if __name__ == '__main__':
    
    #loss_plot()
    get_output()
    compute_dist_matrix()
    #build_nearest_neighbor_hist()









