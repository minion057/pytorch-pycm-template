from torchvision.transforms import *

import torch
import torch.nn.functional as F

class SquarePad_Last: # LM
    def __call__(self, pic):
        if type(pic) != torch.Tensor: raise TypeError('Only torch.Tensor can come.')
        s=pic.size()
        H, W = s[-2], s[-1]
        max_len = max(H, W) 
        vertical_padding = max_len - H
        horizontal_padding = max_len - W
        padding = (0, 0, 0, vertical_padding) if horizontal_padding == 0 else (0, horizontal_padding)
        return F.pad(pic, padding, mode='constant', value=0)

class SquarePad_Side: # CNN
    def __call__(self, pic):
        if type(pic) != torch.Tensor: raise TypeError('Only torch.Tensor can come.')
        s=pic.size()
        H, W = s[-2], s[-1]
        max_len = max(H, W) 
        vertical_padding = (max_len - H) // 2
        horizontal_padding = (max_len - W) // 2
        padding = (0, 0, vertical_padding, vertical_padding) if horizontal_padding == 0 else (horizontal_padding, horizontal_padding)
        return F.pad(pic, padding, mode='constant', value=0)