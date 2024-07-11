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
    content = convert_keys_to_string(content)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_dict2json(content, file_path):
    content = convert_keys_to_string(content)
    with open(file_path, 'w', encoding='utf-8') as handle:
        json.dump(content, handle, ensure_ascii=False, indent='\t')

def convert_keys_to_string(d):
    if not isinstance(d, dict): return d
    new_dict = {}
    for key, value in d.items():
        # Convert key to string
        new_key = str(key)
        
        # Recursively apply to nested dictionaries
        if isinstance(value, dict): 
            new_value = convert_keys_to_string(value)
        elif isinstance(value, list): 
            new_value = [convert_keys_to_string(item) if isinstance(item, dict) else item for item in value]
        else: new_value = value
        
        new_dict[new_key] = new_value
    return new_dict
    
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
    if type(item) in [list, np.ndarray]:      
        if all([0, 1] == item_class): return True
        if all(i in classes for i in item_class): return False
    else: return False
    return True

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
    label_classes = np.unique(label)
    item = label[0]
    
    if check_onehot_label(item, classes): # Is onehot
        label2label = np.argmax(np.array(label), axis=1) 
    elif np.array(classes).dtype == label_classes.dtype:
        # Class and label are the same type as label, so get the index.
        label2label = [classes.index(a) for a in label]
    elif label_classes.dtype in [np.int32, np.int64] and all(label_classes == np.array(list(range(len(label_classes))))):
        # Label is a number and has elements from 0 to the length of label_classes, so use it as an index.
        # Use np.int32, np.int64 instead of np.integer for future compatibility.
        label2label = np.array(label)
    else: raise ValueError(f'Unable to convert label to integer type.\nCurrent classes: {classes}, items in label: {label_classes}.')
    return label2label