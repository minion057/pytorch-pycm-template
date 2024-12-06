import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from utils import convertOneHotEncoding
from libauc import losses

def ce_loss(output, target, classes=None, device=None, **kwargs):
    return F.cross_entropy(output, target, **kwargs)
def bce_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1): 
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        output = F.softmax(output, dim=1)[:, 1]  # Assuming class 1 is the positive class
        return F.binary_cross_entropy(output, target.float(), **kwargs)
    return F.binary_cross_entropy_with_logits(output.squeeze(), target.to(device, dtype=torch.double), **kwargs)

def focal_loss(output, target, classes=None, device=None, **kwargs):
    reduction = kwargs.get('reduction', 'mean')
    alpha = kwargs.get('alpha', None)
    gamma = kwargs.get('gamma', .2)
    if reduction not in ['mean', 'sum']: raise ValueError(f"Invalid reduction mode: {reduction}")
    if gamma is None: raise ValueError("Focal Loss requires 'gamma' to be set.")
    
    ce_loss = F.cross_entropy(output, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1-pt)**gamma * ce_loss)
    if alpha is not None:
        alpha = alpha[target]
        focal_loss = alpha * focal_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
def binary_focal_loss(output, target, classes=None, device=None, **kwargs): # only for binary classification
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('SigmoidFocalLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return sigmoid_focal_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)

def soft_margin_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=False, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('SoftMarginLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return F.soft_margin_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)

def hinge_embedding_loss(output, target, classes=None, device=None, **kwargs):
    if output.shape != target.shape and (len(output.shape) != 1 and len(target.shape) == 1):
        target = convertOneHotEncoding(target, classes, useBinaryConversion=False, useMultiConversion=False)
        if output.shape[-1] == 1: output = output.squeeze()
        elif output.shape[-1] != 2: 
            raise ValueError('HingeEmbeddingLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        else: output = output[:, 1] - output[:, 0]
    return F.hinge_embedding_loss(output.squeeze(), target.to(dtype=torch.float, device=device), **kwargs)

def binary_auc_marging_loss(output, target, classes=None, device=None, **kwargs):
    # Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification (2021, ICCV)
    # AUC-Margin loss with squared-hinge surrogate loss for optimizing AUROC
    # Must be used with the PESG optimiser.
    # It is recommended to train first with CE LOSS and then use it for secondary training with BEST model.
    margin = kwargs.get('margin', 1.0)
    version = kwargs.get('version', 'v1')
    target = convertOneHotEncoding(target, classes, useBinaryConversion=True, useMultiConversion=False)
    loss_fn = losses.AUCMLoss(margin=margin, version=version, device='cpu' if device is None else device)
    return loss_fn(output, target)
def auc_marging_loss(output, target, classes=None, device=None, **kwargs):
    # Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification (2021, ICCV)
    # AUC-Margin loss with squared-hinge surrogate loss for optimizing AUROC
    # Must be used with the PESG optimiser.
    # It is recommended to train first with CE LOSS and then use it for secondary training with BEST model.
    margin = kwargs.get('margin', 1.0)
    loss_fn = losses.MultiLabelAUCMLoss(margin=margin, num_labels=len(classes), device='cpu' if device is None else device)
    return loss_fn(output, target)