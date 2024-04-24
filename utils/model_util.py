import torch
import numpy as np
from collections import OrderedDict

import timm # for pre-trained model


def get_mean_std(data):    
    # calculate mean over each channel (e.g., r,g,b)
    channel = data.shape[1]
    mean, std = [], []
    for idx in range(channel):
        mean.append(data[:,idx,:,:].mean())
        std.append( data[:,idx,:,:].std())
    return mean, std

def cal_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return model.__str__() + '\nTrainable parameters: {}'.format(params)


def get_layers(model):
    layers = OrderedDict()
    for key, module in model.named_modules():
        if key == '': continue
        key_split = key.split('.')
        parents = ('.'.join([(f'[{k}]' if k.isdigit() else k) for k in key_split[:-1]])).replace('.[', '[')
        if parents != '':
            child = key_split[-1]
            key = f'{parents}[{child}]' if child.isdigit() else f'{parents}.{child}'
        if len(layers) != 0 and parents in layers.keys(): layers.pop(parents)
        layers[key] = module
    return layers

def register_forward_hook_layer(model, module_hook_func, layer_idx=None, pre:bool=False):
    """
    Register a function as a hook to be executed before performing forward pass at a specific layer.
    
    :param module_hook_func: The function to be executed in the hook.
    :param layer_idx: Index of the specified layer.
    :param pre: If true, the hook is registered using `register_forward_pre_hook`.
                Otherwise, it is registered using `register_forward_hook`.
    
    :return: The result of the hook. (Remove the hook as needed.)
    """
    if layer_idx is None: handle = model.register_forward_pre_hook(module_hook_func)
    else:
        for idx, (layer_name, module) in enumerate(get_layers(model).items()):
            if idx == layer_idx:
                handle = eval(f'model.{layer_name}.register_forward_pre_hook(module_hook_func)') if pre \
                            else eval(f'model.{layer_name}.register_forward_hook(module_hook_func)')
                print(f'{"Pre " if pre else ""}Hook... {layer_name} - {module}')
                break
        else: raise ValueError(f'Invalid layer Index: {layer_idx}')
    return handle

def load_timm_model(model_name, pretrained:bool=True, num_classes=None):
    # https://huggingface.co/docs/timm/en/reference/models#timm.create_model
    ''' Example Code    
    if pretrained:
        # https://huggingface.co/timm/convnextv2_atto.fcmae_ft_in1k
        num_classes, in_chans, _ = change_kwargs(**kwargs)
        model = load_timm_model('convnextv2_atto.fcmae_ft_in1k', pretrained=True, num_classes=num_classes)
        if in_chans is not None: model.stem[0] = nn.Conv2d(in_chans, model.stem[0].out_channels, 
                                                           kernel_size=model.stem[0].kernel_size, stride=model.stem[0].stride)
    '''
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)   

def load_download_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    key_list = list(checkpoint.keys())
    if 'model' in key_list: return checkpoint['model']
    elif 'state_dict' in key_list: return checkpoint['state_dict']
    return checkpoint

def load_url_checkpoint(path):
    # https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url
    checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu', check_hash=True)
    key_list = list(checkpoint.keys())
    if 'model' in key_list: return checkpoint['model']
    elif 'state_dict' in key_list: return checkpoint['state_dict']
    return checkpoint
    

def change_kwargs(**kwargs):
    num_classes, in_chans = None, None
    if 'num_classes' in list(kwargs.keys()):
        num_classes = kwargs['num_classes']
        kwargs.pop('num_classes')
    if 'in_chans' in list(kwargs.keys()):
        in_chans = kwargs['in_chans']
        kwargs.pop('in_chans')
    return num_classes, in_chans, kwargs