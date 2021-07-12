import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from network import resnet_50_embedding
from torch.optim import lr_scheduler
from torch import optim
from dataset import LimeCardinalTripletDataset
from torch.nn.modules.loss import TripletMarginLoss
from train import train

exp_name = os.path.splitext(os.path.basename(__file__))[0]
print('Experiment name: {}'.format(exp_name))
resume_dict = None
lr = 1e-3
batch_size = 6
print('Load dataset')
image_transform = transforms.Compose([transforms.CenterCrop((700, 1000)),
                                      transforms.RandomCrop((700, 700)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
dataset = LimeCardinalTripletDataset('./terraref/scanner3DTop/lime_cardinal_dataset/dataset/', transform=image_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
print('Init model')
model = resnet_50_embedding()
loss_func = TripletMarginLoss()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
train(model, dataloader, loss_func, optimizer, scheduler, 30,
      resume_dict=resume_dict, ckp_dir='./checkpoints/{}'.format(exp_name))
