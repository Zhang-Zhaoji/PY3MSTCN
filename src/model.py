import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.Sequential(*[\
            DilatedResidualLayer(2 ** _, num_f_maps, num_f_maps) for _ in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask)->torch.Tensor:
        x = self.conv_1x1(x)
        x = self.layers(x, mask)
        return self.conv_out(x) * mask[:,0:1,:] # so what is this mask?

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out) 
        out = self.dropout(out)
        return (x+out) * mask[:,0:1, :] # why?

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([SingleStageModel(num_layers, num_f_maps, num_classes, num_classes) for _ in range(num_stages-1)])
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:,0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


def MSTCN_criterion(output:torch.Tensor, target:torch.Tensor, mask:torch.Tensor)->torch.Tensor:
    loss = F.cross_entropy(output.transpose(1, 2), target, ignore_index=-100)
    loss += 0.15 * torch.mean(torch.clamp(F.mse_loss(F.log_softmax(output[:, :, 1:], dim=1), F.log_softmax(target[:, :, :-1], dim=1), reduction='none'), min=0.0, max=16.0) * mask[:, :, 1:])
    return loss


def build_from_cfg(cfg:Config) -> nn.Module:
    model = MultiStageModel(cfg.num_stages, cfg.num_layers, cfg.num_f_maps, cfg.dim, cfg.num_classes)
    if cfg.resume:
        model = torch.load(cfg.resume)
    return model