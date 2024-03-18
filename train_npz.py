import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0

import argparse
import collections
import torch
import numpy as np

from torchinfo import summary
from torchviz import make_dot
import model.model as module_arch

from parse_config import ConfigParser
import data_loader.npz_loaders as module_data
import model.loss as module_loss
import model.metric_curve_plot as module_curve_metric
import model.metric as module_metric

from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    train_data_loader = data_loader.loaderdict['train'].dataloader
    valid_data_loader = data_loader.loaderdict['valid'].dataloader

    # build model architecture, then print to console
    classes = train_data_loader.dataset.classes
    config['arch']['args']['num_classes'] = len(classes)
    config['arch']['args']['in_channels'], config['arch']['args']['H'], config['arch']['args']['W'] = data_loader.size
    model = config.init_obj('arch', module_arch)
    if config['arch']['visualization']:
        logger.debug('Save the model graph...\n')
        graph_path = config.output_dir / config['arch']['type']
        logger.debug(graph_path)
        modelviz = config.init_obj('arch', module_arch)
        make_dot(modelviz(next(iter(train_data_loader))[0]), params=dict(list(modelviz.named_parameters())), show_attrs=True, show_saved=True).render(graph_path, format='png')
        del modelviz

    # print the model infomation
    # 1. basic method
    # logger.info(model)
    # 2. to use the torchinfo library (from torchinfo import summary)
    input_size = next(iter(train_data_loader))[0].shape
    logger.info('\nInput_size: {}'.format(input_size))
    model_info = str(summary(model, input_size=input_size, verbose=0))
    logger.info('{}\n'.format(model_info))
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    curve_metric = [getattr(module_curve_metric, met) for met in config['curve_metrics']] if 'curve_metrics' in config.config.keys() else None
    if curve_metric is None: print('curve_metric is not set.\n')

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if lr_scheduler is None: print('lr_scheduler is not set.\n')

    trainer = Trainer(model, criterion, metrics, curve_metric, optimizer,
                      config=config,
                      classes=classes,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


""" Run """
args = argparse.ArgumentParser(description='PyTorch pycm Template')
args.add_argument('-c', '--config', default=None,  type=str,  help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None,  type=str,  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None,  type=str,  help='indices of GPUs to enable (default: all)')
args.add_argument('-t', '--test',   default=False, type=bool, help='Whether to enable test mode (default: False)')

# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)

main(config)
