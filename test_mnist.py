import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0

import argparse
import collections
import torch
import numpy as np

import data_loader.mnist_data_loaders as module_data
from torchvision import transforms
import data_loader.transforms as module_transforms
import model.loss as module_loss
import model.metric_curve_plot as module_curve_metric
import model.metric as module_metric

import model.model as module_arch
from parse_config import ConfigParser
from trainer import Tester
from utils import prepare_device


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    trsfm = None
    if 'trsfm' in config['data_loader']['args'].keys():
        tf_list = []
        for k, v in config['data_loader']['args']['trsfm'].items():
            if v is None: tf_list.append(getattr(module_transforms, k)())
            else: tf_list.append(getattr(module_transforms, k)(**v))
        trsfm = transforms.Compose(tf_list) 
    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers'],
        trsfm=trsfm
    )

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    
    # print the model infomation
    # 1. basic method
    # logger.info(model)
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
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    curve_metric = [getattr(module_curve_metric, met) for met in config['curve_metrics']] if 'curve_metrics' in config.config.keys() else None
    
    tester = Tester(model, criterion, metrics, curve_metric,
                      config=config,
                      classes=test_data_loader.classes,
                      device=device,
                      data_loader=test_data_loader)

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
