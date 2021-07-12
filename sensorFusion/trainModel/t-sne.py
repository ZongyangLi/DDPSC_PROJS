'''
Created on Aug 11, 2020

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
import torch.utils.data as torch_u_data
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_tsne(X, plot=True):
    tsne_model = TSNE(n_components=2, perplexity=40, learning_rate=100, verbose=10)
    tsne_Y = tsne_model.fit_transform(tsne_vectors)
    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.scatter(tsne_Y[:, 1], tsne_Y[:, 0], c=tsne_labels, s=1, cmap='hsv')
        
from datetime import date
import utils
from dataset import OPENFusionStereoDataset, OPENFusionJointDataset
batch_size = 20

image_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor()])
#image_transform = transforms.ToTensor()
class OPENFusionStereoDatasetPath(OPENFusionJointDataset):
    def __getitem__(self, index):
        return super().__getitem__(index) + (self.image_paths[index], )
# dataset = OPENScanner3dDatasetPath('./datasets/OPEN', start_date=start_date, end_date=end_date, exclude_date=exclude_date,
#                                                       transform=image_transform, exclude_cultivar=exclude_cultivar)
#image_transform = transforms.Compose([transforms.ToTensor()])
'''
dataset = OPENFusionStereoDatasetPath('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/rgb_crop_resized', transform=image_transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=4)
'''

'''
train_dataset_1 = OPENFusionStereoDatasetPath('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/rgb_crop_resized_test', transform=image_transform)
train_dataset_2 = OPENFusionStereoDatasetPath('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/thermal_crop_resized_test', transform=image_transform)
list_of_datasets = [train_dataset_1, train_dataset_2]
dataloader = torch.utils.data.DataLoader(torch_u_data.ConcatDataset(list_of_datasets), batch_size=batch_size, 
                                         shuffle=False, num_workers=4)
                                         '''
train_dataset = OPENFusionStereoDatasetPath('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_test', transform=image_transform)
#fusion_dateset = TripletNearDateJointDataset(train_dataset, class_by='plot', date_by='scan_date', neighbor_distance=neighbor_distance, collate_fn=None)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model_state_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss/20200915_221501874/epoch_11.pth')
model = network.resnet_50_embedding()
#model.load_state_dict(model_state_dict['model_state_dict'])

# inference
model.load_state_dict(model_state_dict['model_state_dict'])
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
vectors = torch.cat(vector_list, 0)
labels = {}
for l in label_list:
    for key, val in l.items():
        if key not in labels:
            labels[key] = []
        labels[key].extend(val)
image_paths = np.concatenate(image_path_list)

vectors = torch.cat(vector_list, 0)

labels = {}
for l in label_list:
    for key, val in l.items():
        if key not in labels:
            labels[key] = []
        labels[key].extend(val)
        
image_paths = np.concatenate(image_path_list)

vectors = torch.cat(vector_list, 0).cpu().numpy()
labels = {}
for l in label_list:
    for key, val in l.items():
        if key not in labels:
            labels[key] = []
        labels[key].extend(val)
image_paths = np.concatenate(image_path_list)

task_name = 'NearDateJointTripletMarginLoss_resized300'
load_result_ep = 11
torch.save({'vectors': vectors,
            'labels': labels,
            'image_paths': image_paths}, './results/{}_ep_{}.pth'.format(task_name, load_result_ep))


result_dict = torch.load('./results/{}_ep_{}.pth'.format(task_name, load_result_ep))
vectors = result_dict['vectors']
labels = pd.DataFrame(result_dict['labels'])
image_paths = result_dict['image_paths']

tsne_vectors = vectors
tsne_labels = labels

tsne_model = TSNE()
tsne_Y = tsne_model.fit_transform(tsne_vectors)

image_paths_modified = ['./' + os.path.join(*i.split('/')[-3:]) for i in image_paths]
image_paths_modified[0]

vis_df = pd.DataFrame(tsne_labels)
vis_df['tsne_x'] = tsne_Y[:, 1]
vis_df['tsne_y'] = tsne_Y[:, 0]
vis_df['plot_id'] = vis_df['plot']
vis_df['scan_date'] = vis_df['scan_date'].astype(int)
vis_df['image_path'] = image_paths_modified

vis_df.to_csv(f'./{task_name}_ep{load_result_ep}.csv')




































