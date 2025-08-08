import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name = None):
        self.reset()
        if name is not None:
            self.name = name
        else:
            self.name = "DefaultAverageMeter"

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}:{self.avg:.6f} = {self.sum:.6f}/{self.count:.6f}'
    
def _count_iou(pred_label:torch.Tensor, _cls:int, cls_gt:torch.Tensor)->torch.Tensor:
    """ We adopt a slightly different Implementation of IOU from Causal-Effect Traffic Accident Dataset here
, due to the difference of the ground truth format.

    params:
    pred_label: torch.Tensor, shape: [B, T], B for batch size, T for time steps = 208;
    pred_label is the computed max_label, which is the class index with the highest probability for each time step.

    _cls: int, class index, 1 for cause and 2 for effect;

    cls_gt: torch.Tensor, shape: [B, T], B for batch size, T for time steps = 208;
    cls_gt is the ground truth label, which is 1 for cause and 2 for effect.
    """
    pred_eq_cls = pred_label == _cls
    cls_gt_eq_cls = cls_gt == _cls
    inter = torch.sum(pred_eq_cls & cls_gt_eq_cls, dim=1)
    union = torch.sum(pred_eq_cls|cls_gt_eq_cls, dim=1)
    iou = inter / (union + 1e-8)
    return iou

def compute_exact_iou(output:torch.Tensor, cls_gt:torch.Tensor, temporal_mask:torch.Tensor, predtype:str='both') -> torch.Tensor|tuple[torch.Tensor, torch.Tensor]: 
    """
    output: [B, C, T], prediction logits
    cls_gt: [B, T], 0 for background, 1 for foreground
    temporal_mask: [B, T], 1 for valid, 0 for invalid
    """
    pred_label = torch.argmax(output, dim=1)
    valid_label = pred_label * temporal_mask
    if predtype == 'both':
        return _count_iou(valid_label, 1, cls_gt), _count_iou(valid_label, 2, cls_gt)
    elif predtype == 'cause':
        return _count_iou(valid_label, 1, cls_gt)
    elif predtype == 'effect':
        return _count_iou(valid_label, 2, cls_gt)

def compute_temporalIoU(iou_set:list[torch.Tensor]) -> torch.Tensor:
    """
    analyze a series of IOU, compute the amount of each class prediction iou over some threshold
    params:
    iou_set: list of torch.Tensor, each tensor is the iou of a class over all samples
    return:
    cnt: torch.Tensor, shape: [9], the amount of each class prediction iou over some threshold,
    the threshold is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cnt[0] is the amount of iou over 0.1, cnt[1] is the amount of iou over 0.2, and so on.
    """
    cnt = torch.zeros(9)
    for bi in range(0,len(iou_set)):
        for thr in range(0,10):
            if iou_set[bi] > thr/10:
                cnt[thr-1] += 1
    cnt /= len(iou_set)
    return cnt

