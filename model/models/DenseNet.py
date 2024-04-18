""" ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
- Papaer: https://arxiv.org/pdf/1608.06993.pdf
- Official Code: https://github.com/liuzhuang13/DenseNet

- The models used in the paper are all.
"""

import torch.nn as nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201

def DenseNet121(num_classes:int=1000, pretraind:bool=False):    
    model = densenet121(weights='DenseNet121_Weights.DEFAULT') if pretraind else densenet121(num_classes=num_classes)
    if pretraind and num_classes != 1000: model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet161(num_classes:int=1000, pretraind:bool=False):    
    model = densenet161(weights='DenseNet161_Weights.DEFAULT') if pretraind else densenet161(num_classes=num_classes)
    if pretraind and num_classes != 1000: model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet169(num_classes:int=1000, pretraind:bool=False):    
    model = densenet169(weights='DenseNet169_Weights.DEFAULT') if pretraind else densenet169(num_classes=num_classes)
    if pretraind and num_classes != 1000: model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet201(num_classes:int=1000, pretraind:bool=False):    
    model = densenet201(weights='DenseNet201_Weights.DEFAULT') if pretraind else densenet201(num_classes=num_classes)
    if pretraind and num_classes != 1000: model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model