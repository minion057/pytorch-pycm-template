from base import BaseHook

import numpy as np
import torch
from utils import show_mix_result, close_all_plots

class CutMix(BaseHook):
    def __init__(self, beta:float=0.1, prob:float=0.5, writer=None):
        self.type = 'cutmix'
        super().__init__(self.type, cols=['lam', 'rand_index'], writer=writer)
        self.beta, self.prob = beta, prob
        self.init_lam = np.random.beta(beta, beta)

    def lam(self):
        return self._data.loc[self.type, 'lam']
    
    def rand_index(self):
        return self._data.loc[self.type, 'rand_index']
    
    def forward_hook(self, module, input_data, output_data):
        return self.without_hook(output_data)
    
    def forward_pre_hook(self, module, input_data):
        r = np.random.rand(1)
        use_data = input_data[0]
        if use_data.size()[0] > 1:
            if self.prob is None or (self.beta > 0 and r < self.prob): 
                device = use_data.get_device()
                da_result = self._run(use_data.detach().cpu().clone())
                if device != -1: da_result = da_result.cuda()
            # 1. Method for using both original and augmented data.
            if self.prob is None:  return (torch.cat((use_data, da_result), 0), )
            # 2. Method for using only one of the original or augmented data.
            if self.beta > 0 and r < self.prob: return (da_result,)
    
    def without_hook(self, input_data):
        r = np.random.rand(1)
        if input_data.size()[0] > 1:
            if self.prob is None or (self.beta > 0 and r < self.prob): 
                device = input_data.get_device()
                da_result = self._run(input_data.detach().cpu().clone())
                if device != -1: da_result = da_result.cuda()
            # 1. Method for using both original and augmented data.
            if self.prob is None: return torch.cat((input_data, da_result), 0)
            # 2. Method for using only one of the original or augmented data.
            if self.beta > 0 and r < self.prob: return da_result
        else: return input_data
        
    def _run(self, data):
        # Original code: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L229
        size =  data.size()  # B, C, H, W
        B, H, W = size[0], size[-2], size[-1] 
        
        rand_index = np.arange(0, B)
        np.random.shuffle(rand_index)
        self._data.loc[self.type, 'rand_index'] = rand_index

        bbox_W1, bbox_H1, bbox_W2, bbox_H2 = self._rand_bbox(H, W, self.init_lam)
        if bbox_H1==bbox_H2: bbox_H2+=1
        if bbox_W1==bbox_W2: bbox_W2+=1
        
        # adjust lambda to exactly match pixel ratio
        self._data.loc[self.type, 'lam'] = 1 - ((bbox_H2 - bbox_H1) * (bbox_W2 - bbox_W1) / (H*W))

        mix_data = data.detach().clone()
        mix_data[:, :, bbox_H1:bbox_H2, bbox_W1:bbox_W2] = mix_data[rand_index, :, bbox_H1:bbox_H2, bbox_W1:bbox_W2]
        if self.writer is not None and B > 1:
            img_cnt = B if B < 5 else 5
            cut_data = []
            for idx in range(img_cnt):
                cut_data.append([data[idx, 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2],
                                 data[rand_index[idx], 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2],
                                 mix_data[idx, 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2]])
            cut_data = torch.as_tensor(np.array(cut_data))
            self.writer.add_figure(f'input_{self.type}', show_mix_result(cut_data))
            close_all_plots()
        return mix_data

    def _rand_bbox(self, H, W, lam): # for mix methods
        # Original code: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
        # 1. Calculate the size of the patch to be created.      
        patch_ratio = np.sqrt(1. - lam) # Patch ratio
        patch_H, patch_W = int(H * patch_ratio), int(W * patch_ratio)
        
        # 2. Obtain a random value within the size of the original image.
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 3. Extract the coordinates for the patch area.
        bbox_W1 = np.clip(cx - patch_W // 2, 0, W)
        bbox_H1 = np.clip(cy - patch_H // 2, 0, H)
        bbox_W2 = np.clip(cx + patch_W // 2, 0, W)
        bbox_H2 = np.clip(cy + patch_H // 2, 0, H)
        return bbox_W1, bbox_H1, bbox_W2, bbox_H2
    
    def loss(self, loss_ftns, output, target, logit):
        random_index, lam = self.rand_index(), self.lam()
        basic_loss  = loss_ftns(output[:len(target)], target, logit)
        if random_index is None: return {'loss':basic_loss, 'target':target}
        if len(random_index) != len(target): raise ValueError('Target and the number of shuffled indexes do not match.')
        random_loss = loss_ftns(output[len(target):] if self.prob is None else output, target[random_index], logit)
        loss = basic_loss*lam + random_loss*(1.-lam) 
        return {'loss':loss, 'target':torch.cat((target, target[random_index]), 0) if self.prob is None else target}