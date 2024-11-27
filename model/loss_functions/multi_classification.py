import torch.nn.functional as F

# import torch
# from utils import check_onehot_encoding_1
# def nll_loss(output, target, classes=None, device=None, **kwargs):
#     log softmax가 꼭 필요한데, 사용하면 ce loss와 동일함
#     if check_onehot_encoding_1(target[0].cpu(), classes): target = torch.max(target, 1)[-1] 
#     return F.nll_loss(output, target, **kwargs)

def ce_loss(output, target, classes=None, device=None, **kwargs): # cross_entropy_loss
    return F.cross_entropy(output, target, **kwargs)