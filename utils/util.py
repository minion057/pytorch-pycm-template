import numpy as np
import torch
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import RocCurveDisplay

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_dict2json(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as handle:
        json.dump(content, handle, ensure_ascii=False, indent='\t')
        
def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.\n")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def reset_device(mode, show_message:bool=True):
    if mode not in ['memory', 'cache']:
        raise ValueError(f'INVALID mode: "{mode}"')
    if mode == 'memory':
        print('You can delete unnecessary variables.')
    else:
        if show_message:
            print(f'\nThe garbage collector is currently running.')
            print(f'Before cache size : {round(torch.cuda.memory_reserved()/1024**3,1)}GB.')
        torch.cuda.empty_cache()
        if show_message: print(f'After cache size : {round(torch.cuda.memory_reserved()/1024**3,1)}GB.\n')

def tb_projector_resize(data, label_img, features):           
    # probs 추가하고 싶으면 metadata_header, zip list 이용해서 수정
    _, c, h, w = data.shape
    resize_h = h
    while True:
        if resize_h < 30: break
        resize_h = resize_h//2  
    data = torch.nn.functional.interpolate(data, (resize_h), mode='bilinear', align_corners=False, antialias=True) 
    _, c, h, w = data.shape                   
    label_img = torch.cat((label_img, data), 0) if label_img is not None else data
    features = torch.cat((features, data.clone().view(-1, c*h*w)), 0) if features is not None else data.clone().view(-1, c*h*w) #28*28 -> 90MB
    return label_img, features

def check_onehot_label(item, classes):
    item_class = np.unique(np.array(item), return_counts=True)[0]
    if type(item) == int: return False
    elif len(item_class) != len(classes): return False #print('class num')
    elif 0 not in item_class and 1 not in item_class: return False #print(item_class)
    else: return True

def onehot_encoding(label, classes):
    if type(classes) == np.ndarray: classes = classes.tolist() # for FutureWarning by numpy
    item = label[0]
    if not check_onehot_label(item, classes): # label to onehot
        if item not in classes: classes = np.array([idx for idx in range(len(classes))])   
        label, classes = np.array(label), np.array(classes)
        if len(classes.shape)==1: classes = classes.reshape((-1, 1))
        if len(label.shape)==1: label = label.reshape((-1, 1 if type(item) not in [list, np.ndarray] else len(item)))
        oh = OneHotEncoder()
        oh.fit(classes)
        label2onehot = oh.transform(label).toarray()
    else: label2onehot = np.array(label)
    return label2onehot

def integer_encoding(label, classes): #  by index of classes
    if type(classes) == np.ndarray: classes = classes.tolist() # for FutureWarning by numpy
    sorted_classes = deepcopy(classes); sorted_classes.sort()
    item = label[0]
    if check_onehot_label(item, classes) and type(item) in [list, np.ndarray]: label2label = np.argmax(label, axis=1)
    elif all(np.unique(label) != sorted_classes) or sorted_classes[0] != 0: label2label = [classes.index(a) for a in label]
    else: label2label = np.array(label)
    return label2label