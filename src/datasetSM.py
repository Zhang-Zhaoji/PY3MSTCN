# copied from https://github.com/tackgeun/CausalityInTrafficAccident/tree/master
from numpy.random import randint
import argparse, pickle, os, math, random, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import dataset

class CausalityInTrafficAccident(dataset.CausalityInTrafficAccident):
    """Saliency map Causality In Traffic Accident Dataset."""
    
    def __init__(self, p, split, test_mode=False):
        super(CausalityInTrafficAccident, self).__init__(p, split, test_mode, load_data = False)
        DATA_ROOT = 'data'
        self.feature = p['feature']
        self.split = split
        if split == 'train':
            data_length = (0, 1355)
        elif split == 'val':
            data_length = (1355, 1355 + 264)
        elif split == 'test':
            data_length = (1355 + 264, 1355 + 264 + 279)
        p['use_randperm'] = 7802
        self.feat_dir = os.path.join(DATA_ROOT, p['feature_folder'], f'{self.feature}_npy')
        self.feed_type = p['feed_type']

        self.use_flip = False # used to be true

        self.feature_dim = p['input_size']
        self.seq_length = 208
        self.fps = 25
        self.vid_length = self.seq_length * 8 / self.fps
        
        if(self.feed_type == 'classification'):
            self.num_segments = p["num_segments"] # default 3
            self.new_length = p['new_length']
            self.num_causes = 18
            self.num_effects = 7

        self.test_mode = test_mode
        self.random_shift = False

        if('both' in self.feature):
            self.use_flow = True
            self.use_rgb = True
        elif('rgb' in self.feature):
            self.use_flow = False
            self.use_rgb = True
        else:
            print(f'feature type of {self.feature} is not supported, maybe')
            self.use_flow = False
            self.use_rgb = True

        dv = p['dataset_ver']
        self.anno_dir = os.path.join(DATA_ROOT, 'saliency_annotation.pkl')
        
        with open(self.anno_dir, 'rb') as f:
            self.annos = pickle.load(f)
        if(self.use_flow):
            print('Saliency map currently do not support flow features')
            exit(1)
        if(self.use_flip):
            print('Saliency map currently do not support flip')
            exit(1)

        start_idx = data_length[0]
        end_idx = data_length[1]
        self.annos = self.annos[start_idx:end_idx]
        if(p['use_randperm'] > 0):
            torch.manual_seed(p['use_randperm'])
            indices = torch.randperm(len(self.annos))
            self.annos = [self.annos[i] for i in indices]
        else:
            indices = list(range(len(self.annos)))
        self.global_indices = [start_idx + i for i in indices]

        if(self.feed_type == 'detection'):
            self.positive_thres = p['positive_thres']


    def __len__(self):
            return len(self.annos)

    def __getitem__(self, idx):
        if self.feed_type == 'detection':
            return self.feed_detections(idx)
        elif self.feed_type == 'classification':
            return self.feed_classification(idx)
        elif self.feed_type == 'multi-label':
            return self.feed_multi_label(idx)

    def get_feature(self, idx):
        # idx 是本 split 内的下标，需要先映射到全局
        global_idx = self.global_indices[idx]

        rgb_path = os.path.join(self.feat_dir, f'{global_idx}.npy')
        _rgb_feat = np.load(rgb_path)#, mmap_mode='r')
        _rgb_feat = torch.from_numpy(_rgb_feat.astype(np.float32))

        rgb_feat = torch.zeros(self.seq_length, self.feature_dim)
        assert _rgb_feat.size(0) <= self.seq_length
        actual_len = min(_rgb_feat.size(0), self.seq_length)
        rgb_feat[:actual_len, :] = _rgb_feat[:actual_len, :]

        flow_feat = torch.zeros(0)
        return rgb_feat, flow_feat, actual_len
    
def collate_fn(sample):
    feat_list = [s['feature'] for s in sample]
    label_list = [s['label'] for s in sample]
    mask_list = [s['mask'] for s in sample]

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)

    masks = torch.stack(mask_list, dim=0)

    # return {'feature': features, 'label': labels, 'mask': mask}
    return features, labels , masks   
