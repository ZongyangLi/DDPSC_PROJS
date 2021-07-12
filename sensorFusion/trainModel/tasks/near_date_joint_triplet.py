'''
Created on Aug 27, 2020

@author: zli
'''

import os, re
import torch
import numpy as np
from numpy import linspace
from tqdm import tqdm
from torchvision import transforms
import sys
sys.path.append("/home/zli/WorkSpace/PyWork/pytorch-py3/reverse-pheno-master")
from network import resnet_50_embedding
from torch.optim import lr_scheduler
from torch import optim
from loss import TripletMarginLoss, NCALoss, HardNegTripletMarginLoss, HardNegNCALoss, MyNCALoss
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from logger import TensorBoardLogger
from datetime import datetime
from dataset import OPENFusionStereoDataset, OPENFusionJointDataset
from dataset_wrapper import TripletDataset, TripletNearDateDataset, TripletNearDateJointDataset
from sampler import TripletPosetNearDateBatchSampler, TripletPosetNearDateJointDataBatchSampler
from glob import glob
from scipy.stats.mstats_basic import moment
import torch.utils.data as torch_u_data
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataloader, loss_func, optimizer, scheduler, n_epochs, acc_func=None, train_name='default', 
                 resume_dict=None, ckp_dir=None, resume_ep='latest',
                 logger=None, tb_log_dir=None, log_step_interval=100, comment=None, val_dataloader=None):
        '''
        If resume_dict is not None then load from resume_dict and ignore files stored in ckp_dir
        '''
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.acc_func = acc_func
        self.logger = logger
        self.ckp_dir = ckp_dir
        self.tb_log_dir = tb_log_dir
        self.log_step_interval = log_step_interval
        self.start_epoch=0
        self.epoch_loss = None
        self.epoch_acc = None
        self.epoch_num = None
        self.step_loss = None
        self.step_acc = None
        self.step_num = None
        self.val_loss = None
        self.val_nn_acc = None
        self.steps_per_epoch = len(self.dataloader)
        if self.ckp_dir is None:
            self.ckp_dir = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/{}/{}'.format(train_name, self.timestamp)
        if not os.path.isdir(self.ckp_dir):
            os.makedirs(self.ckp_dir)
        # find model
        ckp_dir_file_names = os.listdir(self.ckp_dir)
        if model is None:
            if 'jit_model.pth' in ckp_dir_file_names:        
                self.model = torch.jit.load(os.path.join(self.ckp_dir, 'jit_model.pth'))
            else:
                raise ValueError('No model found')
        # find resume ep
        if resume_ep == 'latest':
            epoch_file_list = glob(os.path.join(self.ckp_dir, 'epoch_*.pth'))
            if len(epoch_file_list):
                prog = re.compile("epoch_([0-9]+).pth")
                epoch_number_list = [int(prog.findall(i)[0]) for i in epoch_file_list]
                resume_ep = max(epoch_number_list)
                print(f'Latest epoch {resume_ep} found.')
            else:
                resume_ep = None
        else:
            assert resume_ep.isnumeric(), ValueError("resume_ep must be numeric or 'latest'.")
            assert os.path.isfile(os.path.join(self.ckp_dir, f"epoch_{resume_ep}.pth")), \
                   FileNotFoundError(f"epoch {resume_ep} not found in {self.ckp_dir}")
            resume_ep = int(resume_ep)
        if resume_dict is not None:
            self.load_epoch(resume_dict)
        elif resume_ep is not None:
            self.load_epoch(os.path.join(self.ckp_dir, f"epoch_{resume_ep}.pth"))
        # Logger
        if self.logger is None:
            if self.tb_log_dir is None:
                self.tb_log_dir = 'tb_logs/{}'.format(train_name)
            self.tb_log_dir = os.path.join(self.tb_log_dir, self.timestamp)
            self.logger = TensorBoardLogger(self.tb_log_dir)
        self.logger.log_init(self)
        self.record_params = ['loss_func', 'step_loss', 'step_acc', 'val_loss', 'val_nn_acc']
        self.epoch_finish_hook = []

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            self.epoch_num = epoch
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            print('---- start epoch: {}/{}\tlearning rate:{:.2E} ----'.format(self.epoch_num, self.n_epochs, current_lr))
            self.epoch_loss, self.epoch_acc = self.train_epoch()
            self.val_loss, self.val_nn_acc = self.validation_epoch()
            for fn in self.epoch_finish_hook:
                fn(self)
            self.scheduler.step()
            print('end epoch: {}/{}\ttrain loss: {:.2f}\tvalidation loss: {}\tvalidation nn accuracy: {}\n'.format(self.epoch_num, self.n_epochs, self.epoch_loss, self.val_loss, self.val_nn_acc))
            self.save_epoch()
            
    def validation_epoch(self):
        val_step_loss = 0
        vector_list = []
        label_list = []
        
        self.model.eval()
        tbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        for batch_idx, (data, target) in tbar:
            self.step_num = batch_idx
            self.optimizer.zero_grad()
            data = tuple(data)
            if type(data) is tuple:
                data = tuple(d.cuda() for d in data)
                model_output = [self.model(d) for d in data]
            elif type(data) is torch.Tensor:
                model_output = self.model(data.cuda())
            else:
                raise TypeError(f'Unknown type of input data{type(data)}')
            loss = self.loss_func(model_output, target)
            val_step_loss += loss.item()
            
            for (o,t) in zip(model_output,target):
                o = torch.nn.functional.normalize(o, p=2, dim=1)
                c = o.cpu().squeeze(0).detach().numpy()
                vector_list.append(c)
                label_list.append(t)
            #print(val_step_loss)
        val_loss = val_step_loss/len(self.val_dataloader)
        labels = {}
        for l in label_list:
            for key, val in l.items():
                if key not in labels:
                    labels[key] = []
                labels[key].extend(val)
        #vectors_norm = [torch.nn.functional.normalize(d, p=2, dim=1) for d in vector_list]
        vectors = vector_list
        
        # compute nearest neighbor accuracy
        total_nums = len(vectors)
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
            
            check_flag = False
            for ind in range(len(min_index)):
                val = min_index[ind]
                pair_label = labels['plot'][val]
                if not check_flag:
                    if (base_label == pair_label) and (base_sensor != pair_sensor):
                        total_correct += 1
                        check_flag = True
                    
        val_nn_acc = float(total_correct)/total_nums
        print('total correct: {}\n'.format(total_correct))
        
        return val_loss, val_nn_acc
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0            
        tbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for batch_idx, (data, target) in tbar:
            self.step_num = batch_idx
            self.optimizer.zero_grad()
            data = tuple(data)
            if type(data) is tuple:
                data = tuple(d.cuda() for d in data)
                model_output = [self.model(d) for d in data]
            elif type(data) is torch.Tensor:
                model_output = self.model(data.cuda())
            else:
                raise TypeError(f'Unknown type of input data{type(data)}')
            loss = self.loss_func(model_output, target)
            self.step_loss = loss.item()
            total_loss += self.step_loss
            loss.backward()
            self.optimizer.step()
            self.logger.step(self)
            if self.acc_func is None:
                tbar.set_description('loss: {:.2f}'.format(self.step_loss))
            else:
                acc = self.acc_func(model_output, target)
                self.step_acc = acc
                total_acc += self.step_accq
                tbar.set_description('loss: {:.2f}, acc: {:.2f}'.format(self.step_loss, acc))
        epoch_loss = total_loss/len(self.dataloader)
        epoch_acc = None
        if self.acc_func is not None:
            epoch_acc = total_acc/len(self.dataloader)
        return epoch_loss, epoch_acc
        
    def loss_plot(self):          
        tbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        date_list = []
        loss_list = []
        ind = 0
        max_num = 3000
        for batch_idx, (data, target) in tbar:
            ind += 1
            if ind > max_num:
                break
            data = tuple(data)
            if type(data) is tuple:
                data = tuple(d.cuda() for d in data)
                model_output = [self.model(d) for d in data]
            elif type(data) is torch.Tensor:
                model_output = self.model(data.cuda())
            else:
                raise TypeError(f'Unknown type of input data{type(data)}')
            loss = self.loss_func(model_output, target)
            loss_list.append(loss.item())
            date_list.append(target[0]['scan_date'].item())
            
        # draw plot
        total_nums = len(loss_list)
        
        min_date = min(date_list)
        max_date = max(date_list)

        hist = np.zeros(max_date-min_date+1)
        hist_count = np.zeros(max_date-min_date+1)
        
        for i in range(total_nums):
            ind_date = date_list[i]-min_date
            loss = loss_list[i]
            hist[ind_date] = hist[ind_date] + loss
            hist_count[ind_date] = hist_count[ind_date] + 1
                
        np_out = hist/hist_count
        #np_out[ np_out==0 ] = np.nan
        x = linspace(0, max_date-min_date, max_date-min_date+1)
        y = np_out
        plt.plot(x,y, marker=".", markersize=40)
        plt.ylim(0, 1)
        plt.show()
        
        return
        
            
            
        
    def add_epoch_hook(self, func):
        self.epoch_finish_hook.append(func)
        return len(self.epoch_finish_hook) - 1
    
    def remove_epoch_hook(self, i):
        self.epoch_finish_hook.pop(i)
        
    def save_jit_model(self):
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, os.path.join(self.ckp_dir, 'jit_model.pth'))

    def save_epoch(self):
        save_path = os.path.join(self.ckp_dir, 'epoch_{}.pth'.format(self.epoch_num))
        torch.save({'epoch': self.epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()},
        save_path)

    def load_epoch(self, resume_dict):
        if type(resume_dict) is str:
            resume_dict = torch.load(resume_dict)
        self.start_epoch = resume_dict['epoch'] + 1
        self.model.load_state_dict(resume_dict['model_state_dict'])
        self.optimizer.load_state_dict(resume_dict['optimizer'])
        self.scheduler.load_state_dict(resume_dict['scheduler'])


# backward capability
def train(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=None, ckp_dir=None, train_name=None,
          tb_log_dir=None, tb_log_step_interval=100, val_dataloader=None):
    trainer = Trainer(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=resume_dict, train_name=train_name,
                      ckp_dir=ckp_dir, tb_log_dir = tb_log_dir, log_step_interval=tb_log_step_interval, val_dataloader=val_dataloader)
    trainer.train()
    #trainer.loss_plot()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    resume_dict = None
    #resume_dict = torch.load('/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/NearDateJointTripletMarginLoss_RotateOtherWay90_Flip/20201209_114516818/epoch_29.pth')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    train_name='NearDateJointTripletMarginLoss_RotateDiffWay33_temp3.5_double'
    tb_log_dir = '/home/zli/WorkSpace/PyWork/pytorch-py3/reverse-pheno-master/tasks/tb_logs/{}'.format(train_name)
    ckp_dir = '/media/zli/Seagate Backup Plus Drive/trained_models/pytorch-py3/checkpoints/{}/{}'.format(train_name,timestamp)
    print('Training on large dataset')
    lr = 1e-1
    batch_size = 24
    val_batch_size = 1
    #class_size = 3
    neighbor_distance=3
    n_epochs = 40
    print('Load dataset')
    
    
    image_transform = transforms.Compose([transforms.RandomRotation(degrees=33),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.CenterCrop(320),
                                          transforms.ToTensor()])
    
    val_image_transform = transforms.Compose([transforms.CenterCrop(320),
                                          transforms.ToTensor()])
    '''
    
    image_transform = transforms.Compose([transforms.RandomRotation(degrees=20),
                                          transforms.CenterCrop(320),
                                          transforms.Resize((224,224)),
                                          transforms.ToTensor()])
    '''
    
    '''
    train_dataset = OPENFusionStereoDataset('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/rgb_crop_resized', transform=image_transform)
    #fusion_dateset = TripletNearDateDataset(dataset, class_by='plot', date_by='scan_date', neighbor_distance=4, collate_fn=None)
    batch_sampler = TripletPosetNearDateBatchSampler(train_dataset, class_by='plot', date_by = 'scan_date', batch_size=batch_size, 
                                                     class_size=class_size, neighbor_distance=neighbor_distance, shuffle_date=True)
    #dataloader = torch.utils.data.DataLoader(fusion_dateset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
    '''
    
    train_dataset = OPENFusionJointDataset('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_double/', transform=image_transform)
    fusion_dateset = TripletNearDateJointDataset(train_dataset, class_by='plot', date_by='scan_date', neighbor_distance=neighbor_distance, collate_fn=None, transform=val_image_transform)
    dataloader = torch.utils.data.DataLoader(fusion_dateset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_dataset = OPENFusionJointDataset('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_3/joint_training_data_validation_double/', transform=val_image_transform)
    val_fusion_dateset = TripletNearDateJointDataset(val_dataset, class_by='plot', date_by='scan_date', neighbor_distance=neighbor_distance, collate_fn=None, transform=val_image_transform)
    val_dataloader = torch.utils.data.DataLoader(val_fusion_dateset, batch_size=val_batch_size, shuffle=True, num_workers=0)
    
    
    print('Init model')
    model = resnet_50_embedding()
    #loss_func = TripletMarginLoss()
    loss_func = MyNCALoss()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 15, gamma=0.1, last_epoch=-1)
    print('Train 1 epoch')
    train(model, dataloader, loss_func, optimizer, scheduler, n_epochs, resume_dict=resume_dict, tb_log_dir=tb_log_dir, ckp_dir=ckp_dir, train_name=train_name, val_dataloader=val_dataloader)
    
    
    
    
    