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

# The entire size = 1896
# # train 70% validation 15% test 15%
# # (0 ~ 1326) (1326 ~ 1611) (1611 ~ 1896)
# parser.add_argument('--dataset_ver', type=str, default='Mar9th')
# parser.add_argument('--train_start', type=int, default=0)
# parser.add_argument('--train_end', type=int, default=1355)
# parser.add_argument('--val_start', type=int, default=1355)
# parser.add_argument('--val_end', type=int, default=1355+290)
# parser.add_argument('--test_start', type=int, default=1355+290)
# parser.add_argument('--test_end', type=int, default=1355+290+290)

# parser.add_argument('--use_randperm', type=int, default=7802)

# parser.add_argument('--use_flip', type=bool, default=True)

# parser.add_argument('--num_causes', type=int, default=18)
# parser.add_argument('--num_effects', type=int, default=7)

# if(args.dataset_ver == 'Nov3th' or args.dataset_ver == 'Mar9th'):
#     args.train_start = 0
#     args.train_end   = 1355
#     args.val_start   = args.train_end
#     args.val_end     = args.train_end + 290
#     args.test_start  = args.val_end
#     args.test_end    = args.val_end + 290

class CausalityInTrafficAccident(Dataset):
    """Causality In Traffic Accident Dataset."""
    
    def __init__(self, p, split, test_mode=False, load_data = True):
        DATA_ROOT = 'data'
        self.feature = p['feature']
        self.split = split
        if split == 'train':
            data_length = (0, 1355)
        elif split == 'val':
            data_length = (1355, 1355 + 290)
        elif split == 'test':
            data_length = (1355 + 290, 1355 + 290 + 290)
        p['use_randperm'] = 7802

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


        if load_data:
            dv = p['dataset_ver']
            self.anno_dir = os.path.join(DATA_ROOT, f'annotation-{dv}-25fps.pkl')
            # for not conflict with the saliency map code
            with open(self.anno_dir, 'rb') as f:
                self.annos = pickle.load(f)

                feat_rgb = torch.load(os.path.join(DATA_ROOT,p['feature_folder'], f'i3d-rgb-fps25-{dv}.pt'))
                if(self.use_flow):
                    feat_flow= torch.load(os.path.join(DATA_ROOT,p['feature_folder'], f'i3d-flow-fps25-{dv}.pt'))

                if(self.use_flip):
                    feat_rgb_flip = torch.load(os.path.join(DATA_ROOT,p['feature_folder'], f'i3d-rgb-flip-fps25-{dv}.pt'))
                    if(self.use_flow):
                        feat_flow_flip = torch.load(os.path.join(DATA_ROOT,p['feature_folder'], f'i3d-flow-flip-fps25-{dv}.pt'))

                start_idx = data_length[0]
                end_idx = data_length[1]

                if(p['use_randperm'] > 0):
                    torch.manual_seed(p['use_randperm'])
                    indices = torch.randperm(len(self.annos))
                    L = indices.numpy().tolist()
                    indices = indices.tolist()
                    remap = lambda I,arr: [arr[i] for i in I]
                    feat_rgb = remap(indices, feat_rgb)
                    if(self.use_flow):
                        feat_flow = remap(indices, feat_flow)

                    if(self.use_flip):
                        feat_rgb_flip = remap(indices, feat_rgb_flip)
                        if(self.use_flow):
                            feat_flow_flip = remap(indices, feat_flow_flip)

                    self.annos = [self.annos[L[l]] for l in range(0, len(self.annos))]

                self.annos = self.annos[start_idx:end_idx]
                self.feat_rgb = feat_rgb[start_idx:end_idx]
                if(self.use_flow):
                    self.feat_flow = feat_flow[start_idx:end_idx]

                if(self.use_flip):
                    self.feat_rgb_flip = feat_rgb_flip[start_idx:end_idx]
                    if(self.use_flow):
                        self.feat_flow_flip = feat_flow_flip[start_idx:end_idx]



    def __len__(self):
            return len(self.annos)
        
    def compute_ious(self, boxes, gt):
        t1 = self.boxes[0, :, :]
        t2 = self.boxes[1, :, :]

        inter_t1 = torch.clamp(t1, min=gt[0]) # torch.cmax(t1, gt[0])
        inter_t2 = torch.clamp(t2, max=gt[1]) # torch.cmin(t2, gt[1])

        union_t1 = torch.clamp(t1, max=gt[0])
        union_t2 = torch.clamp(t2, min=gt[1])

        _inter = F.relu(inter_t2 - inter_t1)
        _union = F.relu(union_t2 - union_t1) + 1e-5

        return _inter / _union

    def __getitem__(self, idx):
        if self.feed_type == 'detection':
            return self.feed_detections(idx)
        elif self.feed_type == 'classification':
            return self.feed_classification(idx)
        elif self.feed_type == 'multi-label':
            return self.feed_multi_label(idx)

    # def get_feature(self, idx):
    #     if(self.use_flip and random.random() > 0.5):
    #         rgb_feat = self.feat_rgb_flip[idx, :, :]
    #         if(self.use_flow):
    #             flow_feat = self.feat_flow_flip[idx, :, :]
    #         else:
    #             flow_feat = torch.zeros(0)
    #     else:
    #         rgb_feat = self.feat_rgb[idx, :, :]
    #         if(self.use_flow):
    #             flow_feat = self.feat_flow[idx, :, :]
    #         else:
    #             flow_feat = torch.zeros(0)

    #     return rgb_feat, flow_feat      

    def get_feature(self, idx):
        # get feature from file database
        if(self.use_flip and random.random() > 0.5):
            if(self.use_rgb):
                _rgb_feat = self.feat_rgb_flip[idx]
            if(self.use_flow):
                _flow_feat = self.feat_flow_flip[idx, :, :]
        else:
            if(self.use_rgb):
                _rgb_feat = self.feat_rgb[idx]
            if(self.use_flow):
                _flow_feat = self.feat_flow[idx, :, :]
        
        # zero-padding
        if(self.use_rgb):
            rgb_feat = torch.zeros(self.seq_length, self.feature_dim)        
            rgb_feat[0:_rgb_feat.size(0), :] = _rgb_feat
        else:
            rgb_feat = torch.zeros(0)

        if(self.use_flow):
            flow_feat = torch.zeros(self.seq_length, self.feature_dim)
            flow_feat[0:_flow_feat.size(0), :] = _flow_feat
        else:
            flow_feat = torch.zeros(0)

        return rgb_feat, flow_feat, _rgb_feat.size(0)      

    def get_det_labels(self, idx):
        annos = self.annos[idx]
        cause_loc = torch.Tensor([annos[1][1], annos[1][2]])
        effect_loc = torch.Tensor([annos[2][1], annos[2][2]])

        vid_length = (annos[0][2] - annos[0][1])
        cause_loc = cause_loc / vid_length
        effect_loc = effect_loc / vid_length

        iou_cause = self.compute_ious(self.boxes, annos[1][1:3])
        iou_effect = self.compute_ious(self.boxes, annos[2][1:3])

        ious = torch.stack([self.iou_bg, iou_cause, iou_effect], dim=0)
        _, labels = torch.max(ious, dim=0)               

        return cause_loc, effect_loc, ious, labels

    # construct labels for SSD detector
    def feed_detections(self, idx):
        try:
            rgb_feat, flow_feat = self.get_feature(idx)
        except:
            print('exception', idx)
        cause_loc, effect_loc, ious, labels = self.get_det_labels(idx)
        
        return {'feature':rgb_feat, 'label':labels}
        # return rgb_feat, flow_feat, cause_loc, effect_loc, labels, ious

    def _sample_indices(self, num_frames):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, num_frames):
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, num_frames):

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1


    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def feed_classification(self, idx):
        annos = self.annos[idx]
        print(f'annos:{annos}')
        
        if(self.use_flip and random.random() > 0.5):
            rgb_feat = self.feat_rgb_flip[idx]
        else:
            rgb_feat = self.feat_rgb[idx]
        
        num_frames = rgb_feat.size(0)

        cause_label = annos[1][3] - 1# - 1 (no background label)
        print(f'cause_label:{cause_label}')
        effect_label = annos[2][3] - self.num_causes - 1  # - 1 (no background label)
        print(f'effect_label:{effect_label}')

        if not self.test_mode:
            segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(num_frames)
        else:
            segment_indices = self._get_test_indices(num_frames)

        #return self.get(record, segment_indices)
        segment_indices = segment_indices - 1
        print(f'segment_indices:{segment_indices}')

        rgb_feat = rgb_feat[segment_indices, :]
        #label = dict()
        #label['cause'] = annos[1][3]
        #label['effect'] = annos[2][3]

        #feat = dict()
        #feat['cause'] = rgb_feat
        #feat['effect'] = flow_feat

        return {'feature':rgb_feat,'cause_label':cause_label,'effect_label': effect_label}
        #return feat, label


    def feed_multi_label(self, idx):
        
        annos = self.annos[idx]
        vid_name = annos[0]
        # seq_length = self.seq_length
        video_start, video_end = annos[0][1:3]
        vid_length = video_end - video_start

        #########
        # input #
        #########
        # rgb = torch.load(self.root_dir + 'rgb%s.pt' % vid_name).transpose(0,1)
        # rgb_feat = torch.zeros(seq_length, rgb.size(1))
        # rgb_feat[0:rgb.size(0), :] = rgb

        rgb_feat, flow_feat, actual_len = self.get_feature(idx)
        seq_length = actual_len
        # if(self.use_flip and random.random() > 0.5):
        #     if(self.use_rgb):
        #         rgb_feat = self.feat_rgb_flip[idx, :, :]
        #     else:
        #         rgb_feat = torch.zeros(0)

        #     if(self.use_flow):
        #         # flow = torch.load(self.root_dir + 'flow%s.pt' % vid_name).transpose(0,1)
        #         # flow_feat = torch.zeros(seq_length, flow.size(1))
        #         # flow_feat[0:flow.size(0), :] = flow
        #         flow_feat = self.feat_flow_flip[idx, :, :]
        #     else:
        #         flow_feat = torch.zeros(0)
        # else:
        #     if(self.use_rgb):
        #         rgb_feat = self.feat_rgb[idx, :, :]
        #     else:
        #         rgb_feat = torch.zeros(0)

        #     if(self.use_flow):
        #         # flow = torch.load(self.root_dir + 'flow%s.pt' % vid_name).transpose(0,1)
        #         # flow_feat = torch.zeros(seq_length, flow.size(1))
        #         # flow_feat[0:flow.size(0), :] = flow
        #         flow_feat = self.feat_flow[idx, :, :]
        #     else:
        #         flow_feat = torch.zeros(0)

        ##########
        # labels #
        ##########
        # cause_loc = torch.Tensor([annos[1][1], annos[1][2]])/vid_length
        # effect_loc = torch.Tensor([annos[2][1], annos[2][2]])/vid_length
        #causality_loc = torch.Tensor([annos[1][1], annos[1][2], annos[2][1], annos[2][2]])/vid_length
        
        ################################################
        # cause label for attention calibration label 
        ################################################        
        cause_start_time = (annos[1][1])/vid_length*seq_length
        cause_end_time = (annos[1][2])/vid_length*seq_length
        cause_start_idx = math.floor(cause_start_time)#int(round(cause_start_time))
        cause_end_idx = math.ceil(cause_end_time)
        if(cause_end_idx > seq_length):
            cause_end_idx = seq_length
        ################################################
        # effect label for attention calibration label 
        ################################################        
        effect_start_time = (annos[2][1])/vid_length*seq_length
        effect_end_time = (annos[2][2])/vid_length*seq_length
        effect_start_idx = math.floor(effect_start_time)# int(round(effect_start_time))
        effect_end_idx = math.ceil(effect_end_time)# int(round(effect_end_time)) + 1
        if(effect_end_idx > seq_length):
            effect_end_idx = seq_length
        ######################################################
        # cause-effect label for attention calibration label 
        ######################################################
        
        clip_length = 208
        causality_mask = torch.zeros(clip_length).long()
        if(cause_end_idx == effect_start_idx):
            effect_portion = effect_start_idx - effect_start_time
            cause_portion = cause_end_time - math.floor(cause_end_time)
            if(effect_portion > cause_portion):
                effect_start_idx = int(math.floor(cause_end_time))
                cause_end_idx = effect_start_idx
            else:
                cause_end_idx = int(math.floor(cause_end_time)) + 1
                effect_start_idx = cause_end_idx

        #if(self.pred_type == 'both'):
        causality_mask[cause_start_idx:cause_end_idx] = 1
        causality_mask[effect_start_idx:effect_end_idx] = 2
        causality_mask[cause_start_idx] = math.ceil(cause_start_time) - cause_start_idx
        causality_mask[effect_start_idx] = math.ceil(effect_start_time) - effect_start_idx
        causality_mask[math.floor(cause_end_time)] = cause_end_idx - math.floor(cause_end_time)
        causality_mask[math.floor(effect_end_time)] = effect_end_idx - math.floor(effect_end_time)
        causality_mask = torch.clamp(causality_mask, 0, 2)
        # print('rgb_feat.size():') 
        # print(rgb_feat.size()) [208, 1024]
        # print('causality_mask.size():')
        # print(causality_mask.size()) [208]
        # sys.exit(1)
        video_mask = torch.zeros(clip_length)
        video_mask[:seq_length] = 1
        # label = torch.Tensor([annos[1][3], annos[2][3]])
        # return rgb_feat, flow_feat, causality_mask, cause_loc, effect_loc, label, annos[0]
        # else:
        # return rgb_feat, flow_feat, causality_mask, cause_loc, effect_loc
        return {'feature':torch.permute(rgb_feat, (1,0)), 'label':causality_mask, 'mask':video_mask} #[208*1024 & 208] -> [1024, 208] & [208]
    
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