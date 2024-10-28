from base import BaseHook

import numpy as np
import torch
from utils import show_mix_result, close_all_plots

class MixUp(BaseHook):
    def __init__(self, alpha:float=1, prob:float=0.5, writer=None):
        self.type = 'mixup'
        super().__init__(self.type, cols=['lam', 'rand_index'], writer=writer)
        self.alpha, self.prob = alpha, prob
        
    def update(self, batch_size):
        self._data.loc[self.type, 'lam'] = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 or batch_size > 1 else 1.
        rand_index = np.arange(0, batch_size)
        np.random.shuffle(rand_index)        
        self._data.loc[self.type, 'rand_index'] = rand_index if batch_size > 1 else None
        
    def lam(self):
        return self._data.loc[self.type, 'lam']
    
    def rand_index(self):
        return self._data.loc[self.type, 'rand_index']
    
    def forward_hook(self, module, input_data, output_data):
        return self.without_hook(output_data)
    
    def forward_pre_hook(self, module, input_data):
        use_data = input_data[0]
        if use_data.size()[0] > 1:
            device = use_data.get_device()
            if self.prob is None and self.alpha <= 0: self.alpha = 1 # To get unconditionally mixed data.
            da_result = self._run(use_data.detach().cpu().clone())
            if device != -1: da_result = da_result.cuda()
            # 1. Method for using both original and augmented data.
            if self.prob is None: return (torch.cat((use_data, da_result), 0), )
            # 2. Method for using only one of the original or augmented data.
            return (da_result, )
    
    def without_hook(self, input_data):
        if input_data.size()[0] > 1:
            device = input_data.get_device()
            if self.prob is None and self.alpha <= 0: self.alpha = 1 # To get unconditionally mixed data.
            da_result = self._run(input_data.detach().cpu().clone())
            if device != -1: da_result = da_result.cuda()
            # 1. Method for using both original and augmented data.
            if self.prob is None: return torch.cat((input_data, da_result), 0)
            # 2. Method for using only one of the original or augmented data.
            return da_result
        else: return input_data
    
    def _run(self, data):
        # Original code: https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L119
        size =  data.size()  # B, C, H, W
        B = size[0]
        self.update(B)
        
        mix_data = data.detach().clone()
        mix_data = self.lam() * mix_data + (1 - self.lam()) * mix_data[self.rand_index(), :]
        
        if self.writer is not None and self.lam() != 1 and B > 1:
            img_cnt = B if B < 5 else 5
            da_data = []
            for idx in range(img_cnt):
                da_data.append([data[idx], data[self.rand_index(), :][idx], mix_data[idx]])
            da_data = torch.as_tensor(np.array(da_data))
            self.writer.add_figure(f'input_{self.type}', show_mix_result(da_data))
            close_all_plots()
        
        # target_a, target_b = target, target[index]
        # loss: lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
        return mix_data
    
    def loss(self, loss_ftns, output, target, logit):
        random_index, lam = self.rand_index(), self.lam()
        basic_loss  = loss_ftns(output[:len(target)], target, logit)
        if random_index is None: return {'loss':basic_loss, 'target':target}
        if len(random_index) != len(target): raise ValueError('Target and the number of shuffled indexes do not match.')
        random_loss = loss_ftns(output[len(target):] if self.prob is None else output, target[random_index], logit)
        loss = basic_loss*lam + random_loss*(1.-lam)
        return {'loss':loss, 'target':torch.cat((target, target[random_index]), 0) if self.prob is None else target}