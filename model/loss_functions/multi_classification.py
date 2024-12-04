import torch
import torch.nn.functional as F

def ce_loss(output, target, classes=None, device=None, **kwargs):
    return F.cross_entropy(output, target, **kwargs)

def focal_loss(output, target, classes=None, device=None, **kwargs):
    reduction = kwargs.get('reduction', 'mean')
    alpha = kwargs.get('alpha', None)
    gamma = kwargs.get('gamma', .0)
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