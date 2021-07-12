from torch.utils.data.sampler import Sampler
from dataset_wrapper import LabelsQueriable
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractclassmethod
from collections.abc import Mapping, Sequence


    
class TripletPosetNearDateBatchSampler(Sampler):
    '''
    generate batches that have different classes and dates
    
    Parameters
    ----------
    dataset: 
        the dataset to generate batches
    class_by:
        key to use to query class
    date_by:
        key to query date
    batch_size:
        total number of samples in a batch
    class_size:
        total number of samples in a class in the batch, or 'm'
    neighbor_distance
        find in a range [-neighbor_distance, +neighbor_distance] for a single batch
    '''
    
    def __init__(self, dataset, class_by, date_by, batch_size, class_size, neighbor_distance=1, shuffle_date=False):        
        if not isinstance(dataset, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset)))
        self.dataset = dataset
        self.labels = self.dataset.get_labels() 
        if not isinstance(self.labels, Mapping):
            raise ValueError("Labels should be mutlilabels. Since the class and date are both required.")
        if class_by not in self.labels:
            raise ValueError("key 'class_by' not in labels keys")
        if date_by not in self.labels:
            raise ValueError("key 'date_by' not in labels keys")

        self.labels_df = pd.DataFrame(self.labels)
        self.class_by = class_by
        self.date_by = date_by
        self.batch_size = batch_size
        self.class_size = class_size
        assert batch_size % class_size == 0, ValueError('batch_size should be times of class_size')
        self.class_per_batch = batch_size // class_size
        self.neighbor_distance = neighbor_distance
        self.shuffle_date = shuffle_date
        self.labels_df['idx'] = self.labels_df.index
        self.labels_df = self.labels_df[[class_by, date_by, 'idx']]
        self.labels_df.set_index([date_by, class_by], inplace=True)
        self.labels_df.sort_index()
        self.labels_dict = self.labels_df.swaplevel(0, 1).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict() 
        self.date_list = list(self.labels_df.index.get_level_values(0).unique())
        # store the offset for some classes in some days. To link the date gap
        # {(date, class):(before_offset, after_offset)}
        self.gap_offset_dict = {}
        # compute length
        self.num_batch = 0
        for date in self.date_list:
            date_pd = self.labels_df.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            self.num_batch += single_date_num_batch


    def __len__(self):
        return self.num_batch

    def __iter__(self):
        batch_list = []
        batch = []
        date_list = self.date_list.copy()
        if self.shuffle_date:
            np.random.shuffle(date_list)
        for date in date_list:
            # select date pd
            date_pd = self.labels_df.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            if single_date_num_batch == 0:
                continue
            for i in range(single_date_num_batch):
                #batch_classes = single_date_class_list[i: i+self.class_per_batch]
                batch_classes = single_date_class_list[i*self.class_per_batch: (i+1)*self.class_per_batch]
                batch = []
                for c in batch_classes:
                    # query cultivar with near date
                    if (date, c) in self.gap_offset_dict:
                        flex_before_dist, flex_after_dist = self.gap_offset_dict[(date, c)]
                        start_date = date - flex_before_dist
                        end_date = date + flex_after_dist
                    else:
                        start_date = date - self.neighbor_distance
                        end_date = date + self.neighbor_distance
                    inrange = {k:v for k, v in self.labels_dict[c]['idx'].items() if start_date < k < end_date}
                    # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                    # check before
                    flex_before_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) >= date) and date != min(self.labels_dict[c]['idx'].keys()):
                        flex_before_dist += 1
                        start_date = date - flex_before_dist
                        # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict[c]['idx'].items() if start_date <= k <= end_date}
                    # check after
                    flex_after_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) <= date) and date != max(self.labels_dict[c]['idx'].keys()):
                        flex_after_dist += 1
                        end_date = date + flex_after_dist
                        #  inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict[c]['idx'].items() if start_date <= k <= end_date}
                    # store flex result
                    if flex_before_dist != self.neighbor_distance or flex_after_dist != self.neighbor_distance:
                        self.gap_offset_dict[(date, c)] = (flex_before_dist, flex_after_dist)
                    # select to let have each day
                    # not effecient enough
                    # date_iter = list(inrange_df.index.get_level_values(0).unique())
                    # np.random.shuffle(date_iter)
                    # for _ in range(self.class_size):
                    #     batch.append(np.random.choice(inrange_df.query(f"{self.date_by}==")))
                    batch.extend(np.random.choice(list(inrange.values()), size=self.class_size, replace=len(inrange)<self.class_size))
                batch_list.append(batch)
        np.random.shuffle(batch_list)
        return iter(batch_list)

class TripletPosetNearDateJointDataBatchSampler(Sampler):
    '''
    generate batches that have different classes and dates
    
    Parameters
    ----------
    dataset1: 
        the first type of dataset to generate batches
    dataset2:
        the second type of dataset to generate batches
    class_by:
        key to use to query class
    date_by:
        key to query date
    batch_size:
        total number of samples in a batch
    class_size:
        total number of samples in a class in the batch, or 'm'
    neighbor_distance
        find in a range [-neighbor_distance, +neighbor_distance] for a single batch
    '''
    
    def __init__(self, dataset1, dataset2, class_by, date_by, batch_size, class_size, neighbor_distance=1, shuffle_date=False):        
        if not isinstance(dataset1, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset1)))
        self.dataset1 = dataset1
        self.labels1 = self.dataset1.get_labels() 
        if not isinstance(self.labels1, Mapping):
            raise ValueError("Labels should be mutlilabels. Since the class and date are both required.")
        if class_by not in self.labels1:
            raise ValueError("key 'class_by' not in labels keys")
        if date_by not in self.labels1:
            raise ValueError("key 'date_by' not in labels keys")
        
        if not isinstance(dataset2, LabelsQueriable):
            raise NotImplementedError('The {} needs the dataset to be a subclass of LabelsQueriable.'.format(type(dataset2)))
        self.dataset2 = dataset2
        self.labels2 = self.dataset2.get_labels() 
        if not isinstance(self.labels2, Mapping):
            raise ValueError("Labels should be mutlilabels. Since the class and date are both required.")
        if class_by not in self.labels2:
            raise ValueError("key 'class_by' not in labels keys")
        if date_by not in self.labels2:
            raise ValueError("key 'date_by' not in labels keys")

        self.labels_df1 = pd.DataFrame(self.labels1)
        self.class_by = class_by
        self.date_by = date_by
        self.batch_size = batch_size
        self.class_size = class_size
        assert batch_size % class_size == 0, ValueError('batch_size should be times of class_size')
        self.class_per_batch = batch_size // class_size
        self.neighbor_distance = neighbor_distance
        self.shuffle_date = shuffle_date
        self.labels_df1['idx'] = self.labels_df1.index
        self.labels_df1 = self.labels_df1[[class_by, date_by, 'idx']]
        self.labels_df1.set_index([date_by, class_by], inplace=True)
        self.labels_df1.sort_index()
        self.labels_dict1 = self.labels_df1.swaplevel(0, 1).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict() 
        self.date_list1 = list(self.labels_df1.index.get_level_values(0).unique())
        # store the offset for some classes in some days. To link the date gap
        # {(date, class):(before_offset, after_offset)}
        self.gap_offset_dict = {}
        # compute length
        self.num_batch1 = 0
        for date in self.date_list1:
            date_pd = self.labels_df1.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            self.num_batch1 += single_date_num_batch
            
        self.labels_df2 = pd.DataFrame(self.labels1)
        self.class_by = class_by
        self.date_by = date_by
        self.batch_size = batch_size
        self.class_size = class_size
        assert batch_size % class_size == 0, ValueError('batch_size should be times of class_size')
        self.class_per_batch = batch_size // class_size
        self.neighbor_distance = neighbor_distance
        self.shuffle_date = shuffle_date
        self.labels_df2['idx'] = self.labels_df2.index
        self.labels_df2 = self.labels_df2[[class_by, date_by, 'idx']]
        self.labels_df2.set_index([date_by, class_by], inplace=True)
        self.labels_df2.sort_index()
        self.labels_dict2 = self.labels_df2.swaplevel(0, 1).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict() 
        self.date_list2 = list(self.labels_df2.index.get_level_values(0).unique())
        # store the offset for some classes in some days. To link the date gap
        # {(date, class):(before_offset, after_offset)}
        self.gap_offset_dict = {}
        # compute length
        self.num_batch2 = 0
        for date in self.date_list2:
            date_pd = self.labels_df2.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            self.num_batch2 += single_date_num_batch
            
        self.num_batch = self.num_batch1 + self.num_batch2


    def __len__(self):
        return self.num_batch

    def __iter__(self):
        batch_list = []
        
        # build dataset1, put data1 into batch_list
        date_list1 = self.date_list1.copy()
        if self.shuffle_date:
            np.random.shuffle(date_list1)
        for date in date_list1:
            # select date pd
            date_pd = self.labels_df1.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            if single_date_num_batch == 0:
                continue
            for i in range(single_date_num_batch):
                batch_classes = single_date_class_list[i*self.class_per_batch: (i+1)*self.class_per_batch]
                batch = []
                for c in batch_classes:
                    # query cultivar with near date
                    if (date, c) in self.gap_offset_dict:
                        flex_before_dist, flex_after_dist = self.gap_offset_dict[(date, c)]
                        start_date = date - flex_before_dist
                        end_date = date + flex_after_dist
                    else:
                        start_date = date - self.neighbor_distance
                        end_date = date + self.neighbor_distance
                    inrange = {k:v for k, v in self.labels_dict1[c]['idx'].items() if start_date < k < end_date}
                    # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                    # check before
                    flex_before_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) >= date) and date != min(self.labels_dict1[c]['idx'].keys()):
                        flex_before_dist += 1
                        start_date = date - flex_before_dist
                        # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict1[c]['idx'].items() if start_date <= k <= end_date}
                    # check after
                    flex_after_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) <= date) and date != max(self.labels_dict1[c]['idx'].keys()):
                        flex_after_dist += 1
                        end_date = date + flex_after_dist
                        #  inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict1[c]['idx'].items() if start_date <= k <= end_date}
                    # store flex result
                    if flex_before_dist != self.neighbor_distance or flex_after_dist != self.neighbor_distance:
                        self.gap_offset_dict[(date, c)] = (flex_before_dist, flex_after_dist)
                    # select to let have each day
                    # not effecient enough
                    # date_iter = list(inrange_df.index.get_level_values(0).unique())
                    # np.random.shuffle(date_iter)
                    # for _ in range(self.class_size):
                    #     batch.append(np.random.choice(inrange_df.query(f"{self.date_by}==")))
                    batch.extend(np.random.choice(list(inrange.values()), size=self.class_size, replace=len(inrange)<self.class_size))
                batch_list.append(batch)
                
        # build dataset1, put data1 into batch_list
        date_list2 = self.date_list2.copy()
        if self.shuffle_date:
            np.random.shuffle(date_list2)
        for date in date_list2:
            # select date pd
            date_pd = self.labels_df2.query("{} == {}".format(self.date_by, date))
            single_date_class_list = date_pd.index.get_level_values(1).unique().values
            np.random.shuffle(single_date_class_list)
            single_date_num_batch = len(single_date_class_list) // self.class_per_batch
            if single_date_num_batch == 0:
                continue
            for i in range(single_date_num_batch):
                #batch_classes = single_date_class_list[i: i+self.class_per_batch]
                batch_classes = single_date_class_list[i*self.class_per_batch: (i+1)*self.class_per_batch]
                batch = []
                for c in batch_classes:
                    # query cultivar with near date
                    if (date, c) in self.gap_offset_dict:
                        flex_before_dist, flex_after_dist = self.gap_offset_dict[(date, c)]
                        start_date = date - flex_before_dist
                        end_date = date + flex_after_dist
                    else:
                        start_date = date - self.neighbor_distance
                        end_date = date + self.neighbor_distance
                    inrange = {k:v for k, v in self.labels_dict2[c]['idx'].items() if start_date < k < end_date}
                    # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                    # check before
                    flex_before_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) >= date) and date != min(self.labels_dict2[c]['idx'].keys()):
                        flex_before_dist += 1
                        start_date = date - flex_before_dist
                        # inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict2[c]['idx'].items() if start_date <= k <= end_date}
                    # check after
                    flex_after_dist = self.neighbor_distance
                    while np.all(np.array(list(inrange.keys())) <= date) and date != max(self.labels_dict2[c]['idx'].keys()):
                        flex_after_dist += 1
                        end_date = date + flex_after_dist
                        #  inrange_df = self.labels_df.query(f"{self.class_by} == '{c}' and {start_date} <= {self.date_by} <= {end_date}")
                        inrange = {k:v for k, v in self.labels_dict2[c]['idx'].items() if start_date <= k <= end_date}
                    # store flex result
                    if flex_before_dist != self.neighbor_distance or flex_after_dist != self.neighbor_distance:
                        self.gap_offset_dict[(date, c)] = (flex_before_dist, flex_after_dist)
                    # select to let have each day
                    # not effecient enough
                    # date_iter = list(inrange_df.index.get_level_values(0).unique())
                    # np.random.shuffle(date_iter)
                    # for _ in range(self.class_size):
                    #     batch.append(np.random.choice(inrange_df.query(f"{self.date_by}==")))
                    batch.extend(np.random.choice(list(inrange.values()), size=self.class_size, replace=len(inrange)<self.class_size))
                batch_list.append(batch)
                
        np.random.shuffle(batch_list)
        return iter(batch_list)


