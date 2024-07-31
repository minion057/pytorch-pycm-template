import numpy as np
import torch
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from pycm import ConfusionMatrix as pycmCM

def ensure_dir(dirname, exist_ok:bool=False):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=exist_ok)

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

def convert_keys_to_string(content:dict):
    if not isinstance(content, dict): return content
    new_dict = {}
    for key, value in content.items():
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
    item = np.array(item)
    item_class = np.unique(item, return_counts=True)[0]
    if all([0, 1] == item_class): 
        if len(classes) == len(item): return True
        else: raise ValueError('It\'s a one-hot encoding format, but it\'s not the same size as classes.')
    if item.ndim == 1 and (item_class == [0] or item_class == [1]):
        raise ValueError(f'It\'s a one-hot encoding format, but only {item_class} exists.')
    return False

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
    label, classes = np.array(label), list(classes)
    # Processing based on data type
    if label.ndim == 1:
        # Make sure all data is the same value
        label_classes = np.unique(label)
        if len(np.unique(label)) == 1:
            unique_value = label[0]
            if unique_value in classes:
                integer_encoded = np.full(len(label), classes.index(unique_value))
            elif unique_value < len(classes):
                integer_encoded = label
            else:
                raise ValueError(f'The unique value {unique_value} in the data is not present in the class list.')
        elif all([label_class in classes for label_class in label_classes]):
            # Convert label encoding to integer encoding
            integer_encoded = np.array([classes.index(label) for label in label])
        elif all([label_class in list(range(len(classes))) for label_class in label_classes]):    
            # Already an integer index
            integer_encoded = label
        else:
            raise ValueError(f'Values in the data ({label_classes}) are not present in the class list ({classes}).')
    else: # Two-dimensional data is likely to be one-hot encoding
        for l in label:
            if not check_onehot_label(l, classes): 
                raise ValueError('Invalid data type: must be one-dimensional (label/integer) or two-dimensional (one-hot).')
        integer_encoded = np.argmax(label, axis=1)
    return np.array(integer_encoded)


def convert_confusion_matrix_to_list(content): 
    if isinstance(content, dict):
        cm = []
        for key in content.keys():
            row = [content[key][sub_key] for sub_key in content.keys()]
            cm.append(row)
        if len(cm) != len(content):
            raise ValueError('A loss occurred when converting a confusion matrix to an array.')
    elif isinstance(content, list) or isinstance(content, np.ndarray): 
        cm = deepcopy(content)
    else: raise TypeError('The confusion matrix can only be a table (dictionary), numpy array, or list.')
    return cm

def save_pycm_object(cm:pycmCM, save_dir, save_name:str='cm'):
    if type(cm) != pycmCM: print('Warning: Can\'t save because there is no confusion matrix.')
    if 'cm' not in save_name: save_name = f'cm_{save_name}'
    try:
        result = cm.save_obj(str(Path(save_dir)/save_name), save_stat=False, save_vector=True)
        if not result['Status']: raise SaveError('Saving the pycm object failed.')
    except SaveError as e: print(f'Save errors: {str(e)}')
    except Exception as e: print(f'Unexpected errors: {str(e)}')
    
def load_pycm_object(object_path:str):
    if Path(object_path).suffix != '.obj': raise TypeError('Only pycm objects saved as `obj` can be loaded.')
    try: json_obj = read_json(object_path)
    except Exception as e: print(f'Unexpected errors: {str(e)}')
    # 객체에서 classes를 변경했다면, json으로 불러와야지만 확인할 수 있음
    # 바로 pycm 객체로 불러온다면, 변경된 classes가 아닌, 원래 classes로 불러와짐
    # 따라서 변경해주는 작업 진행
    object_classes = [class_name for (class_name, pred_value) in json_obj['Matrix']]
    cm = pycmCM(file=open(object_path, "r"))
    cm.classes = object_classes
    cm.table = {cm.classes[cm_idx]:{cm.classes[idx]:v for idx, v in cm_v.items()} for cm_idx, cm_v in enumerate(cm.table.values())}
    return cm