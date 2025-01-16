import torch
import torch.nn.functional as F
from utils import check_onehot_encoding_1

def mse_loss(output, target, classes=None, device=None, **kwargs):
    if check_onehot_encoding_1(target[0].cpu(), classes): target = torch.max(target, 1)[-1] 
    return F.mse_loss(output, target.to(device).float(), **kwargs)

def smooth_l1_loss(output, target, classes=None, device=None, **kwargs): 
    if check_onehot_encoding_1(target[0].cpu(), classes): target = torch.max(target, 1)[-1] 
    return F.mse_loss(output, target.to(device).float(), **kwargs)