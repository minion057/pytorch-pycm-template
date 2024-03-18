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
