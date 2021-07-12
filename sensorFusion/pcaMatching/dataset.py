import sys
import os
import glob
from PIL import Image
from datetime import datetime
import numpy as np
#import pandas as pd
import torch
import gzip
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from copy import copy
from abc import ABCMeta, abstractclassmethod
from dataset_wrapper import LabelsQueriable

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, labels_set, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in labels_set:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, file_names in sorted(os.walk(d)):
            for file_name in sorted(file_names):
                if has_file_allowed_extension(file_name, extensions):
                    path = os.path.join(root, file_name)
                    item = (path, target)
                    images.append(item)

    return images

class GridMatchDataset(Dataset, LabelsQueriable):
    def __init__(self, input_img_path, grid_pool, imgSize=224, transform=None):
        self.input_img_path = input_img_path
        self.grid_pool = grid_pool
        self.imgSize = imgSize
        self.transform = transform
        self.image_paths, self.labels = self._make_dataset() 
        
    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path = self.input_img_path
        paths = []
        labels = {'x1': [], 'y1': [], 'x2': [], 'y2': []}
        for grid in self.grid_pool:
            paths.append(file_path)
            labels['x1'].append(grid[0])
            labels['y1'].append(grid[1])
            labels['x2'].append(grid[2])
            labels['y2'].append(grid[3])
        return paths, labels
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        image = image.resize((self.imgSize, self.imgSize))
        im1 = image.crop((label['x1'], label['y1'], label['x2'], label['y2']))
        img_w, img_h = image.size
        background = Image.new('RGB', (img_w, img_h), (0, 0, 0))
        background.paste(im1, (label['x1'], label['y1']))
        #background.show('show pair grid image')
        
        if self.transform is not None:
            background = self.transform(background)
        return background, label
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]
    
class SimilarityVisualiztionDataset(Dataset, LabelsQueriable):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.image_file_suffix = '.png'
        self.transform = transform
        self.image_paths, self.labels = self._make_dataset() 
        
    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path_list = glob.glob(os.path.join(self.input_dir, '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'plot': []}
        for file_path in file_path_list:
            basename = os.path.basename(file_path)
            plot = basename[:10]
            paths.append(file_path)
            labels['plot'].append(plot)
        return paths, labels
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENFusionStereoDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        self.open_dataset_root = open_dataset_root
        self.image_file_suffix = '.png'
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = open_dataset_root#os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path_list = glob.glob(os.path.join(self.season_root, '**', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'plot': [], 'scan_date': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            plot = path_split[-2]
            scan_date = datetime.strptime(path_split[-1][:10], '%Y-%m-%d').date().toordinal()
            paths.append(file_path)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        '''
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
            '''
        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]
    
class OPENFusionStereoDataset_selectX(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        self.open_dataset_root = open_dataset_root
        self.image_file_suffix = '.png'
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = open_dataset_root#os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        #file_path_list = glob.glob(os.path.join(self.season_root, '**', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'plot': [], 'scan_date': []}
        
        # exclude image less than X images
        list_dirs = os.listdir(self.season_root)
        count_ = 0
        for d in list_dirs:
            sub_dir = os.path.join(self.season_root, d)
            if not os.path.isdir(sub_dir):
                continue
            sub_file_list = glob.glob(os.path.join(sub_dir, '*{}'.format(self.image_file_suffix)))
            if len(sub_file_list) < 7:
                continue
            for file_path in sub_file_list:
                path_split = file_path.split('/')
                plot = path_split[-2]
                scan_date = datetime.strptime(path_split[-1][:10], '%Y-%m-%d').date().toordinal()
                paths.append(file_path)
                labels['plot'].append(plot)
                labels['scan_date'].append(scan_date)
            count_ += 1
            #print(count_)
            if count_ > 100:
                print(count_)
                break
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        '''
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
            '''
        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENScanner3dDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, file_type='depth',
                 start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        if file_type == 'depth':
            self.image_file_suffix = 'p.png'
        elif file_type == 'reflectance':
            self.image_file_suffix = 'g.png'
        elif file_type == 'xyz':
            self.image_file_suffix = 'xyz.npy.gz'
        else:
            raise ValueError('{} file type not exist.'.format(file_type))
        self.open_dataset_root = open_dataset_root
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', 'scanner3DTop', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'scanner': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            plot = path_split[-3]
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            if 'east' in path_split[-1]:
                scanner = 'east'
            elif 'west' in path_split[-1]:
                scanner = 'west'
            else:
                continue
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
            labels['scanner'].append(scanner)
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]

        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENScanner3dSurfNormDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, 
                 start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        self.image_file_suffix = '.png'
        self.open_dataset_root = open_dataset_root
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', 'scanner3DTop_preprocessed', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'scanner': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            plot = path_split[-3]
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            if 'east' in path_split[-1]:
                scanner = 'east'
            elif 'west' in path_split[-1]:
                scanner = 'west'
            else:
                continue
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
            labels['scanner'].append(scanner)
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]

        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class OPENStereoDataset(Dataset, LabelsQueriable):
    def __init__(self, open_dataset_root, season=4, start_date=None, end_date=None,exclude_cultivar=None, exclude_date=None, transform=None):
        self.open_dataset_root = open_dataset_root
        self.image_file_suffix = '.png'
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_date = exclude_date
        self.exclude_cultivar = exclude_cultivar
        self.transform = transform
        self.season_root = os.path.join(self.open_dataset_root, 'season_{}'.format(self.season))
        self.image_paths, self.labels = self._make_dataset() 

    def _make_dataset(self):
        '''return the list of images, labels with format as (file_path, label)'''
        file_path_list = glob.glob(os.path.join(self.season_root, '**', '**', 'stereoTop', '*{}'.format(self.image_file_suffix)))
        paths = []
        labels = {'cultivar': [], 'plot': [], 'scan_date': [], 'scanner': []}
        for file_path in file_path_list:
            path_split = file_path.split('/')
            cultivar = path_split[-4]
            plot = path_split[-3]
            scan_date = datetime.strptime(path_split[-1][:20], '%Y-%m-%d__%H-%M-%S').date().toordinal()
            paths.append(file_path)
            labels['cultivar'].append(cultivar)
            labels['plot'].append(plot)
            labels['scan_date'].append(scan_date)
        # filter dataset by conditions
        # filter by start end
        if self.start_date is not None and self.end_date is not None:
            start_timestamp = self.start_date.toordinal()
            end_timestamp = self.end_date.toordinal()
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if start_timestamp<t<end_timestamp]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        # filter by exclude
        if self.exclude_date is not None:
            exclude_date = [date.toordinal() for date in self.exclude_date]
            filtered_idx = [i for i, t in enumerate(labels['scan_date']) if t not in exclude_date]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        if self.exclude_cultivar is not None:
            exclude_cultivar = self.exclude_cultivar
            filtered_idx = [i for i, c in enumerate(labels['cultivar']) if c not in exclude_cultivar]
            for key in labels.keys():
                labels[key] = [labels[key][i] for i in filtered_idx]
            paths = [paths[i] for i in filtered_idx]
        return paths, labels
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = {key: val[index] for key, val in self.labels.items()}
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_sample_label(self, sample):
        return sample[1]

class WheatTripletDataset(Dataset):
    def __init__(self, df_path=None, key_file_path = None, img_dir=None, transform = None):
        self.img_dir = img_dir
        self.data_df = pd.read_csv(df_path)
        self.key_file = pd.read_csv(key_file_path)
        self.transform = transform
        self.label_to_indices = {label: np.where(np.array(self.data_df.plot_id) == label)[0] for label in self.key_file['plot_id']}
        self.label_to_indices = {k: v for k, v in self.label_to_indices.items() if len(v)!=0}
        
    def __len__(self):
        return len(self.data_df.index)
    
    def __getitem__(self, index):
        # Anchor Image Data
        anchor_img_folder = self.data_df.folder[index]
        anchor_img_name = self.data_df.img_name[index]
        anchor_label = self.data_df.plot_id[index]
        anchor_img = Image.open(os.path.join(self.img_dir, anchor_img_folder, anchor_img_name))
        
        #  Find Negative Images
        negative_label = np.random.choice(list(set(list(self.label_to_indices.keys())) - set(list([anchor_label]))))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative_img_folder = self.data_df.folder[negative_idx]
        negative_img_name = self.data_df.img_name[negative_idx]
        negative_img = Image.open(os.path.join(self.img_dir, negative_img_folder, negative_img_name))
        
        #  Find Positive Image
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.label_to_indices[anchor_label])
        positive_img_folder = self.data_df.folder[positive_idx]
        positive_img_name = self.data_df.img_name[positive_idx]
        positive_img = Image.open(os.path.join(self.img_dir, positive_img_folder, positive_img_name))
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), (anchor_label, anchor_label, negative_label)
    
class WheatSingleDataset(WheatTripletDataset):
    def __getitem__(self, index):
        img_folder = self.data_df.folder[index]
        img_name = self.data_df.img_name[index]
        label = self.data_df.plot_id[index]
        img = Image.open(os.path.join(self.img_dir, img_folder, img_name))
        if self.transform:
            img = self.transform(img)
        return img, label
    
def random_split_and_augment(my_dataset):
    torch.manual_seed(0)
    train_size = int(0.8 * len(my_dataset))
    trainset, testset = torch.utils.data.random_split(my_dataset, [train_size, len(my_dataset)-train_size])
    trainset.dataset = copy(my_dataset)
    
    trainset.dataset.transform = transforms.Compose(
        [transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(30),
         transforms.ToTensor()])
    
    testset.dataset.transform = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor()])
    return trainset, testset
    
class TripletWheatWithTimeDataset(Dataset):
    def __init__(self, look_up, cult_indx_dict, transform = None):
        self.data_df = look_up
        self.cul_indx_dict = cult_indx_dict
        self.transform = transforms.Compose(
        [transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.ToTensor()])
    
    
    def __len__(self):
        return len(self.data_df.index)
    
    def __getitem__(self, index):
        # Anchor Image Data
        anchor_loc = self.data_df.location[index]
        anchor_img = Image.open(anchor_loc)
        anchor_cul = self.data_df.cultivar[index]
        anchor_dat = self.data_df.date[index]
        
        #  Find Negative Images
        negative_cul = np.random.choice(list(set(self.cult_list) - set([anchor_cul])))
        negative_idx = np.random.choice(self.cul_indx_dict[negative_cul])
        negative_loc = self.data_df.location[negative_idx]
        negative_img = Image.open(negative_loc)
        negative_dat = self.data_df.date[negative_idx]
        
        #  Find Positive Image
        positive_cul = anchor_cul
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.cul_indx_dict[anchor_cul])
        
        positive_loc = self.data_df.location[positive_idx]
        positive_img = Image.open(positive_loc)
        positive_dat = self.data_df.date[positive_idx]
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), (anchor_dat, positive_dat, negative_dat), (anchor_loc, positive_loc, negative_loc)

# weighted loss based on imbalanced data
def get_loss_weight(img_num_list):
    # number of awned:   123571
    # number of awnless: 5236
    # set the reduction ratio for awned sample to 0.03
    
    weight_list = 1/img_num_list.float()
    normalized_weight = weight_list * len(weight_list)/weight_list.sum()
    return normalized_weight