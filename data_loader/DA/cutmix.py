from base import BaseHook

import numpy as np
import torch
from utils import show_mix_result, plot_close

class CutMix(BaseHook):
    def __init__(self, beta:float=0.1, writer=None):
        self.type = 'cutmix'
        super().__init__(self.type, cols=['lam', 'rand_index'], writer=writer)
        self.init_lam = np.random.beta(beta, beta)

    def lam(self):
        return self._data['lam'][self.type]
    def rand_index(self):
        return self._data['rand_index'][self.type]
    
    def forward_hook(self):
        def hook(module, input, output):
            output = self._run(output.detach().clone())
        return hook
    
    def forward_pre_hook(self):
        def hook(module, input):
            input = (self._run(input[0].detach().clone()),)
        return hook
        
    def _run(self, data):
        # Original code: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L229
        size =  data.size()  # B, C, H, W
        H, W = size[-2], size[-1] 
        print(f'data size: {H, W}')
        
        rand_index = np.arange(0, size[0])
        rand_index = np.random.shuffle(rand_index)
        self._data['rand_index'][self.type] = rand_index

        bbox_W1, bbox_H1, bbox_W2, bbox_H2 = self._rand_bbox(H, W, self.self.init_lam)
        if bbox_H1==bbox_H2: bbox_H2+=1
        if bbox_W1==bbox_W2: bbox_W2+=1
        print(f'bbox size: {bbox_W1, bbox_H1, bbox_W2, bbox_H2}')
        
        # adjust lambda to exactly match pixel ratio
        self._data['lam'][self.type] = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H*W))

        mix_data = data.detach().clone()
        mix_data[:, :, bbox_H1:bbox_H2, bbox_W1:bbox_W2] = mix_data[rand_index, :, bbox_H1:bbox_H2, bbox_W1:bbox_W2]
        if writer is not None:
            img_cnt = size[0] if size[0] < 5 else 5
            cut_data = []
            for idx in range(img_cnt):
                cut_data.append([data[idx, 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2],
                                 data[rand_index[idx], 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2],
                                 mix_data[idx, 0, bbox_H1:bbox_H2, bbox_W1:bbox_W2]])
            cut_data = torch.as_tensor(np.array(cut_data))
            self.writer.add_image(f'input_{self.type}', show_mix_result(cut_data))
            plot_close()
        
        # calculate loss
        # output = self.model(data)
        # loss = self.criterion(output, target) * lam + self.criterion(output, random_target) * (1. - lam)
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