import torch
import torch.nn.functional as F
from utils import check_onehot_encoding_1


def bce_loss(output, target, classes=None, device=None, **kwargs): # 이진 분류
    if check_onehot_encoding_1(target[0].cpu(), classes): 
        target = torch.max(target, 1)[-1]
    if len(output.shape) != 1 and output.shape[-1] != 1: 
        output = F.softmax(output, dim=1)[:, 1]  # Assuming class 1 is the positive class
        return F.binary_cross_entropy(output, target.float(), **kwargs)
    return F.binary_cross_entropy_with_logits(output.squeeze(), target.type(torch.DoubleTensor).to(device), **kwargs)

def soft_margin_loss(output, target, classes=None, device=None, **kwargs): # 이진 분류
    if check_onehot_encoding_1(target[0].cpu(), classes): 
        target = torch.max(target, 1)[-1] 
        target[target==0] = -1
    if len(output.shape) != 1 and output.shape[-1] != 1:
        if output.shape[-1] != 2: 
            raise ValueError('SoftMarginLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        output = output[:, 1] - output[:, 0]
    return F.soft_margin_loss(output.squeeze(), target, **kwargs)

def hinge_embedding_loss(output, target, classes=None, device=None, **kwargs): # 이진 분류
    if check_onehot_encoding_1(target[0].cpu(), classes): 
        target = torch.max(target, 1)[-1] 
        target[target==0] = -1
    if len(output.shape) != 1 and output.shape[-1] != 1:
        if output.shape[-1] != 2: 
            raise ValueError('HingeEmbeddingLoss is designed for binary classification tasks only. Multi-class classification is not supported.')
        output = output[:, 1] - output[:, 0]
    return F.hinge_embedding_loss(output.squeeze(), target, **kwargs)