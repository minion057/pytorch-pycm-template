import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def conv_output_size(self, IS, KS, PS:int=0, ST:int=1):
        """
        Calculate the output size of convolution layer. ((H-FH+2P)/strid+1)//2
        
        :param IS:  Input   size (height or width)
        :param KS:  Kernel  size
        :param PS:  Padding size
        :param ST:  Stride  size
        """
        return (IS-KS+PS*2)//ST+1
        
    def pooling_output_size(self, IS, PS):
        """
        Calculate the output size of pooling layer. H//ps
        
        :param IS: Input   size (height or width)
        :param PS: Pooling size
        """
        return IS//PS

    def _get_layers(self):
        layers = OrderedDict()
        for key, module in self.named_modules():
            if key == '': continue
            key_split = key.split('.')
            parents = ('.'.join([(f'[{k}]' if k.isdigit() else k) for k in key_split[:-1]])).replace('.[', '[')
            if parents != '':
                child = key_split[-1]
                key = f'{parents}[{child}]' if child.isdigit() else f'{parents}.{child}'
            if len(layers) != 0 and parents in layers.keys(): layers.pop(parents)
            layers[key] = module
        return layers
    
    def register_forward_hook_layer(self, module_hook_func, layer_idx=None, pre:bool=False):
        """
        Register a function as a hook to be executed before performing forward pass at a specific layer.
        
        :param module_hook_func: The function to be executed in the hook.
        :param layer_idx: Index of the specified layer.
        :param pre: If true, the hook is registered using `register_forward_pre_hook`.
                    Otherwise, it is registered using `register_forward_hook`.
        
        :return: The result of the hook. (Remove the hook as needed.)
        """
        print(module_hook_func)
        if layer_idx is None: handle = self.register_forward_pre_hook(module_hook_func)
        else:
            for idx, (layer_name, module) in enumerate(self._get_layers().items()):
                if idx == layer_idx:
                    handle = eval(f'self.{layer_name}.register_forward_pre_hook(module_hook_func)') if pre \
                             else eval(f'self.{layer_name}.register_forward_hook(module_hook_func)')
                    print(f'{"Pre " if pre else ""}Hook... {layer_name} - {module}')
                    break
            else: raise ValueError(f'Invalid layer Index: {layer_idx}')
        return handle
            
    def remove_hook(self, handle):
        """
        Remove a function as a hook.
        
        :param handle: The result of the hook.
        """
        handle.remove()