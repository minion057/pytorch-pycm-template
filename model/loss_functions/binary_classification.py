import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from utils import convertOneHotEncoding


def bce_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1): 
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        output = F.softmax(output, dim=1)[:, 1]  # Assuming class 1 is the positive class
        return F.binary_cross_entropy(output, target.float(), **kwargs)
    return F.binary_cross_entropy_with_logits(output.squeeze(), target.to(device, dtype=torch.double), **kwargs)

def soft_margin_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('SoftMarginLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return F.soft_margin_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)

def hinge_embedding_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('HingeEmbeddingLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return F.hinge_embedding_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)

def binary_focal_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('SigmoidFocalLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return sigmoid_focal_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)