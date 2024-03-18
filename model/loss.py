import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def ce_loss(output, target): # cross_entropy_loss
    return F.cross_entropy(output, target)

def bce_loss(output, target): # binary_cross_entropy
    # return F.binary_cross_entropy(torch.sigmoid(input), target)
    return F.binary_cross_entropy_with_logits(output, target)
