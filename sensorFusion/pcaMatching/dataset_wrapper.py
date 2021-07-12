import sys
import os
import glob
from PIL import Image
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from copy import copy
from abc import ABCMeta, abstractclassmethod
from collections.abc import Mapping, Sequence
#import pandas as pd

def get_label_to_indice(labels):
    labels_set = set(labels)
    return {label: np.where(np.array(labels) == label)[0] for label in labels_set}

class LabelsQueriable(metaclass=ABCMeta):
    """
    An abstract class for Pairs/Lets selector. The dataset that going to use the Lets warppers needs to inherent this class
    """
    @abstractclassmethod
    def get_labels(self):
        """
        Get all the labels of the dataset
        """
        pass

    def get_sample_label(self, sample):
        """
        Given a sample, return the label of the sample
        E.g.:
            A sample returned from __getitem__() formated as (image, label)
            the function returns the label
        """
        pass


class TripletDataset(Dataset):
    """
    A triplet warpper to a map-style dataset. The warpped dataset should inherent from the LabelwiseQueriable 
    """
    def __init__(self, dataset, select_by_label=None, collate_fn=None):
        self.dataset = dataset
        self.select_by_label = select_by_label
        if not isinstance(dataset, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset)))
        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = TripletDataset.default_collate_fn
        self.labels = self.dataset.get_labels()
        if isinstance(self.labels, Mapping):
            if select_by_label is None:
                raise ValueError('The labels from dataset is a Mapping(Multilabel), but no selected label for P/N select. (select_by_label is None)')
            else:
                self.labels = self.labels[select_by_label]
        self.labels_set = set(self.labels)
        self.label_to_indices = get_label_to_indice(self.labels)

    def __getitem__(self, index):
        anchor = self.dataset[index]
        anchor_label = self.dataset.get_sample_label(anchor)
        if isinstance(anchor_label, Mapping):
            anchor_label = anchor_label[self.select_by_label]
        # find index of positive and negative samples
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.label_to_indices[anchor_label])
        negative_label = np.random.choice(list(self.labels_set - set(list([anchor_label]))))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        return self.collate_fn(anchor, positive, negative)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def default_collate_fn(a, p ,n):
        return (a[0], p[0], n[0]), (a[1], p[1], n[1])


class TripletNearDateDataset(Dataset):
    def __init__(self, dataset, class_by, date_by, neighbor_distance=1, collate_fn=None):        
        self.dataset = dataset
        self.class_by = class_by
        self.date_by = date_by
        self.neighbor_distance = neighbor_distance
        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = TripletNearDateDataset.default_collate_fn
        if not isinstance(dataset, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset)))
        self.labels = self.dataset.get_labels()
        if not isinstance(self.labels, Mapping):
            raise ValueError("Labels should be mutlilabels. Since the class and date are both required.")
        if self.class_by not in self.labels:
            raise ValueError("key 'class_by' not in labels keys")
        if self.date_by not in self.labels:
            raise ValueError("key 'date_by' not in labels keys")
        if not isinstance(self.labels, Mapping):
            raise ValueError("Dataset multilabel required")
        self.labels_df = pd.DataFrame(self.labels)
        self.labels_df['idx'] = self.labels_df.index
        self.labels_df = self.labels_df[[class_by, date_by, 'idx']]
        self.labels_df.set_index([class_by, date_by], inplace=True)
        # orgnize time and cultivar

    def __getitem__(self, index):
        anchor = self.dataset[index]
        anchor_label = self.dataset.get_sample_label(anchor)
        anchor_class = anchor_label[self.class_by]
        anchor_date = anchor_label[self.date_by]
        start_int = (anchor_date - self.neighbor_distance)
        end_int = (anchor_date + self.neighbor_distance)
        # select positive
        positive_in_range_df = self.labels_df.query("{} == '{}'".format(self.class_by, anchor_class))
        in_range_df = positive_in_range_df.query("{} <= {} <= {}".format(start_int, self.date_by, end_int))
        flex_dist = self.neighbor_distance
        while len(in_range_df) == 0:
            flex_dist += 1
            start_int = (anchor_date-flex_dist)
            end_int = (anchor_date+flex_dist)
            in_range_df = positive_in_range_df.query("{} <= {} <= {}".format(start_int, self.date_by, end_int))
        positive_idx = np.random.choice(in_range_df['idx'])
        
        # select negative
        negative_in_range_df = self.labels_df.query("{} != '{}'".format(self.class_by, anchor_class))
        in_range_df = negative_in_range_df.query("{} <= {} <= {}".format(start_int, self.date_by, end_int))
        flex_dist = self.neighbor_distance
        while len(in_range_df) == 0:
            flex_dist += 1
            start_int = (anchor_date-flex_dist)
            end_int = (anchor_date+flex_dist)
            in_range_df = negative_in_range_df.query("{} <= {} <= {}".format(start_int, self.date_by, end_int))
        negative_idx = np.random.choice(in_range_df['idx'])
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        return self.collate_fn(anchor, positive, negative)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def default_collate_fn(a, p ,n):
        return (a[0], p[0], n[0]), (a[1], p[1], n[1])


class PosetPairDataset(Dataset):
    '''Return pairs, small first, then large'''
    def __init__(self, dataset, select_by_label=None, collate_fn=None):
        self.dataset = dataset
        self.select_by_label = select_by_label
        if not isinstance(dataset, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset)))
        self.labels = self.dataset.get_labels()
        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = PosetPairDataset.default_collate_fn
        if isinstance(self.labels, Mapping):
            if select_by_label is None:
                raise ValueError('The labels from dataset is a Mapping(Multilabel), but no selected label for P/N select. (select_by_label is None)')
            else:
                self.labels = self.labels[select_by_label]
        self.labels_set = set(self.labels)
        self.label_to_indices = get_label_to_indice(self.labels)

    def __getitem__(self, index):
        anchor = self.dataset[index]
        anchor_label = self.dataset.get_sample_label(anchor)
        if isinstance(anchor_label, Mapping):
            anchor_label = anchor_label[self.select_by_label]
        negative_label = np.random.choice(list(self.labels_set - set(list([anchor_label]))))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative = self.dataset[negative_idx]
        if anchor_label > negative_label:
            return self.collate_fn(negative, anchor)
        else:
            return self.collate_fn(anchor, negative)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def default_collate_fn(a, b):
        return (a[0], b[0]), (a[1], b[1])


class PosetPairNearDateDataset(Dataset):
    def __init__(self, dataset, date_by, class_by=None, neighbor_distance=1, collate_fn=None, flex_neighbor_dist=False):        
        self.dataset = dataset
        self.class_by = class_by
        self.date_by = date_by
        self.neighbor_distance = neighbor_distance
        self.collate_fn = collate_fn
        self.flex_neighbor_dist = flex_neighbor_dist
        self.flex_dist_dict = {}
        if self.collate_fn is None:
            self.collate_fn = self.default_collate_fn
        if not isinstance(dataset, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset)))
        self.labels = self.dataset.get_labels()
        if not isinstance(self.labels, Mapping):
            raise ValueError("Labels should be mutlilabels. Since the class and date are both required.")
        # if self.class_by not in self.labels:
        #     raise ValueError("key 'class_by' not in labels keys")
        if self.date_by not in self.labels:
            raise ValueError("key 'date_by' not in labels keys")
        if not isinstance(self.labels, Mapping):
            raise ValueError("Dataset multilabel required")
        self.labels_df = pd.DataFrame(self.labels)
        self.labels_df['idx'] = self.labels_df.index
        # self.labels_df = self.labels_df[[class_by, date_by, 'idx']]
        # self.labels_df.set_index([class_by, date_by], inplace=True)
        self.labels_df = self.labels_df[[date_by, 'idx']]
        self.labels_df.set_index([date_by], inplace=True)
        # orgnize time and cultivar

    def __getitem__(self, index):
        anchor = self.dataset[index]
        anchor_label = self.dataset.get_sample_label(anchor)
        # anchor_class = anchor_label[self.class_by]
        anchor_date = anchor_label[self.date_by]
        start_int = (anchor_date - self.neighbor_distance)
        end_int = (anchor_date + self.neighbor_distance)
        # find in flex neighbor record
        if self.flex_neighbor_dist and anchor_date in self.flex_dist_dict:
            start_int = (anchor_date - self.flex_dist_dict[anchor_date])
            end_int = (anchor_date + self.flex_dist_dict[anchor_date])
        # query neighobrs
        in_range_df = self.labels_df.query(f"{start_int} <= {self.date_by} <= {end_int} and {self.date_by} != {anchor_date}")
        if not self.flex_neighbor_dist:
            assert len(in_range_df) != 0, ValueError((f'Data dont have any neighbor {self.date_by} at {anchor_date}'))
        else:
            flex_dist = self.neighbor_distance
            while len(in_range_df) == 0:
                flex_dist += 1
                start_int = (anchor_date - flex_dist)
                end_int = (anchor_date + flex_dist)
                in_range_df = self.labels_df.query(f"{start_int} <= {self.date_by} <= {end_int} and {self.date_by} != {anchor_date}")
                # remember the flex_dist for spec day to reduce next query time
            self.flex_dist_dict[anchor_date] = flex_dist

        negative_idx = np.random.choice(in_range_df['idx'])
        negative = self.dataset[negative_idx]
        negative_date =  self.dataset.get_sample_label(negative)[self.date_by]
        if anchor_date > negative_date:
            return self.collate_fn(negative, anchor)
        else:
            return self.collate_fn(anchor, negative)

        
        # positive_idx = index
        # while positive_idx == index:
        #     positive_idx = np.random.choice(in_range_df.query(f"{self.class_by} == '{anchor_class}'")['idx'])
        # negative_idx = np.random.choice(in_range_df.query(f"{self.class_by} != '{anchor_class}'")['idx'])
        # positive = self.dataset[positive_idx]
        # negative = self.dataset[negative_idx]
        # return self.collate_fn(anchor, positive, negative)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def default_collate_fn(a, b):
        return (a[0], b[0]), (a[1], b[1])