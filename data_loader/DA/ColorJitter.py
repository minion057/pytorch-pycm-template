from torchvision import transforms
from base import BaseHook
import numpy as np
import torch
from utils import show_mix_result, close_all_plots

class ColorJitter(BaseHook):
    def __init__(self, prob:float=0.5, writer=None, **aug_kwargs):
        self.type = 'ColorJitter'
        if aug_kwargs == {}:
            raise ValueError('There are no options available for Colorjitter (torchvision.transforms.ColorJitter).')
        available_options = ['brightness', 'contrast', 'saturation', 'hue']
        if not any(k in available_options for k in aug_kwargs.keys()):
            raise ValueError(f'Not an option available in ColorJitter.\nList of available options: {available_options}')
        for k, v in aug_kwargs.items():
            aug_kwargs[k] = tuple(v) if type(v) == list else float(v)
        super().__init__(self.type, cols=['None'], writer=writer)
        self.prob = prob
        self.augmentation = transforms.Compose([transforms.ColorJitter(**aug_kwargs)])
        # brightness=(0.2,0.8), contrast=(0.2,0.8), saturation=(0.2,0.8), hue=(-0.5, 0.5)
    
    def forward_hook(self, module, input_data, output_data):
        return self.without_hook(output_data)
    
    def forward_pre_hook(self, module, input_data):
        r = np.random.rand(1)
        if self.prob is None or r < self.prob: 
            use_data = input_data[0]
            device = use_data.get_device()
            da_result = self._run(use_data.detach().cpu().clone())
            if device != -1: da_result = da_result.cuda()
        # 1. Method for using both original and augmented data.
        if self.prob is None: return (torch.cat((use_data, da_result), 0), )
        # 2. Method for using only one of the original or augmented data.
        if r < self.prob: return (use_data, )
    
    def without_hook(self, input_data):
        r = np.random.rand(1)
        if self.prob is None or (self.beta > 0 and r < self.prob): 
            device = input_data.get_device()
            da_result = self._run(input_data.detach().cpu().clone())
            if device != -1: da_result = da_result.cuda()
        # 1. Method for using both original and augmented data.
        if self.prob is None: return torch.cat((input_data, da_result), 0)
        # 2. Method for using only one of the original or augmented data.
        if self.beta > 0 and r < self.prob: return da_result
    
    def _run(self, data):
        size =  data.size()  # B, C, H, W
        
        aug_data = data.detach().clone()
        aug_data = self.augmentation(aug_data)
        
        if self.writer is not None:
            img_cnt = size[0] if size[0] < 5 else 5
            da_data = [[data[idx], aug_data[idx]] for idx in range(img_cnt)]
            da_data = torch.as_tensor(np.array(da_data))
            self.writer.add_figure(f'input_{self.type}', show_mix_result(da_data, titles=['Original Data', 'Result']))
            close_all_plots()
        return aug_data
    
    def loss(self, loss_ftns, output, target, logit):
        use_target = torch.cat((target, target), 0) if self.prob is None else target
        return {'loss': loss_ftns(output, use_target, logit), 'target':use_target}