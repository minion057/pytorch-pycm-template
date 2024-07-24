import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import collections
import torch
import numpy as np

import model.model as module_arch
# from torchinfo import summary
from utils import cal_model_parameters

import data_loader.npz_loaders as module_data
from torchvision import transforms
import data_loader.transforms as module_transforms

from parse_config import ConfigParser
from runner import TorchcamExplainer as Explainer
from utils import prepare_device

def get_a_explainset(data_loader, n_samples_per_class:int, classes):
    """ Logic for creating datasets for use in XAI.

    Args:
        data_loader: Data loader to extract data for use in XAI.
        n_samples_per_class (int): The number of data to extract for each class.
        classes: The list of label classes used by the data loader.

    Returns:
        n_samples (dict): A dataset to perfome XAI.
                          - The key value is the index of the class, 
                          - The value is the data to use (Tensor or numpy.ndarray type), labels and paths.
    """
    labels = data_loader.dataset.labels
    if isinstance(labels[0], list): labels = np.where(labels == 1)[-1].tolist() # for one-hot encoding
    
    label_classes = np.unique(labels).tolist()
    index_classes = [classes.index(l) for l in label_classes] if label_classes[-1] in classes else label_classes
    
    class_indices = {class_idx:np.where(labels == label_classes[real_idx])[0].tolist() for real_idx, class_idx in enumerate(index_classes)}
    class_indices = dict(sorted(class_indices.items()))
    
    n_samples = {class_name:{'index':class_index[:n_samples_per_class]} for class_name, class_index in class_indices.items()}
    for class_name, class_content in n_samples.items():
        print(f'Example index list of explaination dataset: {class_name} -> {class_content["index"]}')
        data_label = [data_loader.dataset.__getitem__(_) for _ in class_content["index"]]
        n_samples[class_name]['data'] = [_[0] for _ in data_label]
        n_samples[class_name]['labels'] = [_[1].item() for _ in data_label]
        if data_loader.dataset.data_paths is not None:
            n_samples[class_name]['paths'] = np.take(data_loader.dataset.data_paths, class_content["index"], axis=0)
    return n_samples      

def main(config):
    # cpu 코어 제한
    if config.processor_cores is not None: torch.set_num_threads(config.processor_cores)
    
    # setup data_loader instances
    if 'trsfm' in config['data_loader']['args'].keys():
        tf_list = []
        for k, v in config['data_loader']['args']['trsfm'].items():
            if v is None: tf_list.append(getattr(module_transforms, k)())
            else: tf_list.append(getattr(module_transforms, k)(**v))
        config['data_loader']['args']['trsfm'] = transforms.Compose(tf_list)  
    config.config['data_loader']['args']['mode'] = ['test']
    config.config['data_loader']['args']['batch_size'] = 1
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.loaderdict['test'].dataloader
    
    classes = test_data_loader.dataset.classes
    n_samples_per_class = 5
    explainset = get_a_explainset(test_data_loader, n_samples_per_class, classes)
    print(f"Input Shape: {explainset[0]['data'][0].shape}")
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    
    # print the model infomation
    # 1. basic method
    # if model.__str__().split('\n')[-1] != ')': print(model) # Based on the basic model (BaseModel).
    # else: print(cal_model_parameters(model))
    # 2. to use the torchinfo library (from torchinfo import summary)
    # input_size = next(iter(test_data_loader))[0].shape
    # print('\nInput_size: {}'.format(input_size))
    # model_info = str(summary(model, input_size=input_size, verbose=0))
    # print('{}\n'.format(model_info))

    # prepare for (multi-device) GPU training
    if config['n_gpu'] > 1: raise TypeError('To proceed with XAI, please set the GPU to use only one.')
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    
    # CAM requires complete fully connected layer information. 
    # However, there are cases where the model does not have a fully connected layer. 
    # Therefore, run without the CAM method in activation_based_methods.
    # SSCAM adds noise and consumes a lot of VRAM, so run it except when using a GPU.
    gradient_based_methods = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM'] # Fast. 
    activation_based_methods = ['ScoreCAM', 'SSCAM', 'ISCAM'] if device.type == 'cpu' else ['ScoreCAM', 'ISCAM'] # Slow.
    all_methods = gradient_based_methods + activation_based_methods
    exp = Explainer(model, 
                    config=config,
                    classes=classes,
                    device=device,
                    explainset=explainset,
                    explain_methods=all_methods)
    exp.explain(save_type='all') # exp.explain(xai_layers=xai_layers)


""" Run """
args = argparse.ArgumentParser(description='PyTorch pycm Template')
args.add_argument('-c', '--config', default=None, type=str,  help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str,  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,  help='indices of GPUs to enable (default: all)')
args.add_argument('-p', '--core',   default=None, type=int, help='Amount of CPU cores to limit (default: None)')
args.add_argument('-t', '--test',   default=True, type=bool, help='Whether to enable test mode (default: True)')

# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)

main(config)
