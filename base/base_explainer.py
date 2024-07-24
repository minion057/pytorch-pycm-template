import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from utils import ensure_dir, get_layers

class BaseExplainer:
    def __init__(self, model, config, classes, device, xai_layers:list=None):
        self.config = config
        self.device = device
        
        # load architecture params from checkpoint.
        self.model = model
        self.test_epoch = 1
        self.output_dir_name = 'explanation'
        self.output_dir = Path(config.output_dir) / self.output_dir_name / f'epoch{self.test_epoch}'
        # self.output_dir = Path(config.explanation_dir) / f'epoch{self.test_epoch}'
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
            self.output_dir = Path(config.output_dir) / self.output_dir_name / f'epoch{self.test_epoch}'
        else: print("Warning: Pre-trained model is not use.\n")
        
        # Set up your data and model layers for XAI.
        self.classes = classes
        
        self.all_layers = get_layers(self.model)
        self.last_conv_layer_name = self._find_last_conv_layer()
        self.xai_layers = xai_layers
        # if xai_layers is None or len(xai_layers) >= len(self.all_layers): self.xai_layers = self._find_last_conv_layer()
        
        # Setting the save directory path
        if not self.output_dir.is_dir(): ensure_dir(self.output_dir)        
        
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
    
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.test_epoch = checkpoint['epoch']
        
        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            print("Warning: Architecture configuration given in config file is different from that of checkpoint.")
            print("This may yield an exception while state_dict is being loaded.")
        if isinstance(self.model, DP) or isinstance(self.model, DDP): self.model.module.load_state_dict(checkpoint['state_dict'])
        else: self.model.load_state_dict(checkpoint['state_dict'])

        print(f"Checkpoint loaded. Testing from epoch {self.test_epoch}")
        
    @abstractmethod
    def explain(self):
        """
        Logic for performing XAI techniques such as Grad-CAM.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _save_output(self):
        raise NotImplementedError
    
    
    def _freeze_and_get_layers(self, model, target_layers:list=None):
        block_layers_none, max_parts, num_parts_to_remove = False, 0, 1
        while True:
            if block_layers_none: 
                error_message = 'No place found to change requires_grad to True.\n'
                error_message += 'Performed once, and each layer has no child layers.\n'
                error_message += 'No target_layers found, and block not found, so cannot freeze.'
                raise ValueError(error_message)
            if max_parts != 0 and num_parts_to_remove >= max_parts:
                error_message = 'No place found to change requires_grad to True.\n'
                error_message += 'Checked up to the top block, but no target layers found.\n'
                raise ValueError(error_message)
            
            grad_enabled = False # Parameter to check if any part has requires_grad set to True. If False, continue the iteration.
            for layer_name, param in model.named_parameters():
                parts = layer_name.split('.') # Manage block hierarchies as lists by separating them from child hierarchies.
                # 1. Maximum number of child hierarchies
                if max_parts < len(parts): max_parts = len(parts)
                
                # 2. Make sure each layer has no child layers.
                if '.' not in layer_name: block_layers_none = True
                else: block_layers_none = False
                
                # 3. Run
                # Name of the block part to check in the current hierarchy.
                real_layer_name = '.'.join(parts[:-num_parts_to_remove])
                block_check = any(t in real_layer_name for t in target_layers)
                if num_parts_to_remove > 1: block_check = any('.'.join(t.split('.')[:-(num_parts_to_remove-1)]) in real_layer_name for t in target_layers)
                # Verify that the part to check exists.
                if target_layers is not None and real_layer_name in target_layers: # Exact match
                    grad_enabled = True
                    param.requires_grad_(True)
                elif target_layers is not None and block_check: # Block match
                    grad_enabled = True
                    param.requires_grad_(True)
                else: param.requires_grad_(False)
            if target_layers is not None and not grad_enabled: num_parts_to_remove += 1
            else: break
        return model
    
    def _find_last_conv_layer(self):
        last_conv = None
        for layer_name, module in dict(reversed(list(self.all_layers.items()))).items():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = layer_name
                break
        if last_conv is None: raise ValueError('No convolutional layer found in the model!')
        last_conv = last_conv.replace('[', '.').replace(']', '')
        return last_conv
    
    def _find_last_fc_layer(self):
        last_fc = None
        for layer_name, module in dict(reversed(list(self.all_layers.items()))).items():
            if isinstance(module, torch.nn.Linear):
                last_fc = layer_name
                break
        if last_fc is None: raise ValueError('No fully connected layer found in the model!')
        last_fc = last_fc.replace('[', '.').replace(']', '')
        return last_fc