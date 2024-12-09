import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import collections
import torch
import numpy as np

from torchinfo import summary
import model.model as module_arch

from torchvision import transforms
import data_loader.transforms as module_transforms
import data_loader.npz_loaders as module_data
import model.loss as module_loss
import model.plottable_metrics  as module_plottable_metric
import model.metric as module_metric

from parse_config import ConfigParser
from runner import Tester
from utils import prepare_device
from utils import cal_model_parameters


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    if 'trsfm' in config['data_loader']['args'].keys():
        tf_list = []
        for k, v in config['data_loader']['args']['trsfm'].items():
            if v is None: tf_list.append(getattr(module_transforms, k)())
            else: tf_list.append(getattr(module_transforms, k)(**v))
        config['data_loader']['args']['trsfm'] = transforms.Compose(tf_list)  
    config.config['data_loader']['args']['batch_size'] = 1
    try:
        is_test = True
        config.config['data_loader']['args']['mode'] = ['test']
        data_loader = config.init_obj('data_loader', module_data)
        test_data_loader = data_loader.loaderdict['test'].dataloader
    except:
        is_test = False
        logger.warning('No test mode dataset. Thus, validation mode will be used.')
        config.config['data_loader']['args']['mode'] = ['valid']
        data_loader = config.init_obj('data_loader', module_data)
        test_data_loader = data_loader.loaderdict['valid'].dataloader

    # build model architecture, then print to console
    classes = test_data_loader.dataset.classes
    model = config.init_obj('arch', module_arch)
    
    # print the model infomation
    # 1. basic method
    # if model.__str__().split('\n')[-1] != ')': logger.info(model) # Based on the basic model (BaseModel).
    # else: logger.info(cal_model_parameters(model))
    # 2. to use the torchinfo library (from torchinfo import summary)
    input_size = next(iter(test_data_loader))[0].shape
    logger.info('\nInput_size: {}'.format(input_size))
    model_info = str(summary(model, input_size=input_size, verbose=0))
    logger.info('{}\n'.format(model_info))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1: model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics'].keys()]
    plottable_metric = None
    if 'plottable_metrics' in config.config.keys():
        plottable_metric = [getattr(module_plottable_metric, met) for met in config['plottable_metrics'].keys()]

    tester = Tester(model, criterion, metrics, plottable_metric,
                    config=config,
                    classes=classes,
                    device=device,
                    data_loader=test_data_loader,
                    is_test=is_test)

    tester.test()


""" Run """
args = argparse.ArgumentParser(description='PyTorch pycm Template')
args.add_argument('-c', '--config', default=None, type=str,  help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str,  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,  help='indices of GPUs to enable (default: all)')
args.add_argument('-t', '--test',   default=True, type=bool, help='Whether to enable test mode (default: True)')

# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)

main(config)
