import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path

from utils import get_layers


class BaseExplainer:
    def __init__(self, model, config, classes, device, xai_layer_indices:list=None):
        self.config = config
        self.device = device
        
        # Set up your data and model layers for XAI.
        self.classes = classes
        
        self.all_layers = get_layers(self.model)
        if xai_layer_indices is None or max(xai_layer_indices) >= len(self.all_layers): self.xai_layer = self._find_last_conv_layer()
        self.xai_layer = xai_layer_indices
        
        # load architecture params from checkpoint.
        self.model = model
        self.test_epoch = 1
        output_dir_name = 'explanation'
        self.output_dir = Path(config.output_dir) / output_dir_name / f'epoch{self.test_epoch}'
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
            self.output_dir = Path(config.output_dir) / output_dir_name / f'epoch{self.test_epoch}'
        else: print("Warning: Pre-trained model is not use.\n")
        
        # Setting the save directory path
        if not self.output_dir.is_dir(): self.output_dir.mkdir(parents=True)
        
        # Freeze the model
        for param in self.model.parameters(): param.requires_grad_(False)
        
        
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
    
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.test_epoch = checkpoint['epoch'] + 1
        
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
    def _get_a_explainset(self, data_loader, n_samples_per_class):
        """
        For your data loader, import n_samples_per_class of data and labels to perform XAI on.
        
        :param data_loader: your data loader.
        :param n_samples_per_class: How much data to get from each class.
        """
        raise NotImplementedError
    
    def _find_last_conv_layer(self): # 수정하세요
        last_conv = None
        for layer_name, module in dict(reversed(list(self.all_layers.items()))).items():
            if type(module) == torch.nn.Conv2d: 
                last_conv = layer_name
                break
        if last_conv is None: raise ValueError('No convolutional layer found in the model!')
        return [last_conv]