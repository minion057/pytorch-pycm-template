import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import collections
import torch
import numpy as np
from pathlib import Path

from torchinfo import summary
from torchviz import make_dot
import model.model as module_arch

from torchvision import transforms
import data_loader.transforms as module_transforms
import data_loader.npz_loaders as module_data
import data_loader.data_augmentation as module_DA
import model.optim as module_optim
import model.lr_scheduler as module_lr_scheduler
import model.loss as module_loss
import model.plottable_metrics  as module_plottable_metric
import model.metric as module_metric

from parse_config import ConfigParser
from runner import Trainer, FixedSpecTrainer
from utils import prepare_device, reset_device, cal_model_parameters
from utils import read_json, write_json, set_common_experiment_name

from libauc import losses

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def change_config_for_auclossmode():
    args = argparse.ArgumentParser(description='PyTorch pycm Template')
    args.add_argument('-co', '--config_origin', default=None, type=str, help='config file path (default: None)')
    args, unknown = args.parse_known_args()
    config_dict = read_json(args.config_origin)
    
    resume_path = Path(config_dict['trainer']['save_dir']) / 'models' / set_common_experiment_name(config_dict)
    best_model_list = sorted(resume_path.glob('**/model_best.pth'))
    if len(best_model_list) == 0: raise ValueError('There is no best model.')
    elif len(best_model_list) > 1: raise ValueError(f'There are more than one best model. List: {best_model_list}.')
    
    config_dict['loss'] = 'auc_marging_loss'
    config_dict['optimizer']['type'] = 'PESG'
    config_dict['optimizer']['args'] = {'lr':config_dict['optimizer']['args']['lr']}
    
    append_str = '_AUCLOSS'
    # if config_dict['trainer']['save_dir'][-1] != '/': config_dict['trainer']['save_dir'] +=  append_str
    # else: config_dict['trainer']['save_dir'] = config_dict['trainer']['save_dir'].rstrip('/') + append_str
    config_dict['name'] += append_str
    config_save_path = str(args.config_origin).replace('.json', f'{append_str}.json')
    config_dict['AUC LOSS'] = str(best_model_list[-1])
    write_json(config_dict, config_save_path)
    return best_model_list[-1], config_save_path

def init_args():
    args = argparse.ArgumentParser(description='PyTorch pycm Template')
    args.add_argument('-f', '--fixedspectrainer',  default=False,  type=bool, help='Whether to enable fixedspectrainer mode (default: True)')
    args.add_argument('-c',  '--config',        default=None,  type=str,  help='config file path (default: None)')
    args.add_argument('-co', '--config_origin', default=None,  type=str,  help='config file path for auc loss mode (default: None)') 
    args, unknown = args.parse_known_args()
    return args.fixedspectrainer, not args.config_origin is None
    
def parsing_args(config_new_path:str=None):
    args = argparse.ArgumentParser(description='PyTorch pycm Template')
    args.add_argument('-co', '--config_origin', default=None,  type=str,  help='config file path for auc loss mode (default: None)') 
    args.add_argument('-c',  '--config',        default=config_new_path,  type=str,  help='config file path (default: None)')
    args.add_argument('-r',  '--resume',        default=None,  type=str,  help='path to latest checkpoint (default: None)')
    args.add_argument('-d',  '--device',        default=None,  type=str,  help='indices of GPUs to enable (default: all)')
    args.add_argument('-t',  '--test',          default=False, type=bool, help='Whether to enable test mode (default: False)')
    return args

def main(config):
    logger = config.get_logger('train')
    
    # setup data_loader instances
    if 'trsfm' in config['data_loader']['args'].keys():
        tf_list = []
        for k, v in config['data_loader']['args']['trsfm'].items():
            if v is None: tf_list.append(getattr(module_transforms, k)())
            else: tf_list.append(getattr(module_transforms, k)(**v))
        config['data_loader']['args']['trsfm'] = transforms.Compose(tf_list)
    data_loader = config.init_obj('data_loader', module_data)
    train_data_loader = data_loader.loaderdict['train'].dataloader
    valid_data_loader = data_loader.loaderdict['valid'].dataloader

    # build model architecture, then print to console
    classes = train_data_loader.dataset.classes
    model = config.init_obj('arch', module_arch)

    # print the model infomation
    if model.__str__().split('\n')[-1] != ')': logger.info(model) # Based on the basic model (BaseModel).
    else: logger.info(cal_model_parameters(model))
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1: model = torch.nn.DataParallel(model, device_ids=device_ids)

    if IS_AUCLOSS: # model load using best model
        logger.info('\nAUC LOSS MODE. USING MODEL PATH: {}\n'.format(MODEL_BEST_PATH))
        checkpoint = torch.load(MODEL_BEST_PATH, map_location=device, weights_only=False)
        if len(device_ids) > 1: model.module.load_state_dict(checkpoint['state_dict'])
        else: model.load_state_dict(checkpoint['state_dict'])

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics'].keys()]
    plottable_metric = None
    if 'plottable_metrics' in config.config.keys():
        plottable_metric = [getattr(module_plottable_metric, met) for met in config['plottable_metrics'].keys()]
    
    # get function handles of da
    if 'data_augmentation' in config.config.keys():
        da = config['data_augmentation']
        if 'type' not in da.keys(): raise ValueError('Data augmentation type is not set.')
        if 'hook_args' in da.keys():
            if isinstance(da['hook_args'], dict): 
                if any(k not in da['hook_args'].keys() for k in ['pre', 'layer_idx']): 
                    raise ValueError('There is no pre-hook information for DA.')
        else: raise ValueError('There is no hook information for DA.')
        da_ftns = getattr(module_DA, da['type'])
        if 'args' in da.keys():
            if 'prob' in da['args'].keys() and da['args']['prob'] is None:
                if 'sampler' in config['data_loader']['args']:
                    sampling_type = config['data_loader']['args']['sampler']['args']['sampler_type']
                    sampling_type = sampling_type.split('-')[0].split('_')[0][0].upper() # Oversampling -> O, Undersampling -> U
                    if sampling_type == 'O':
                        logger.warning('Although the data loader has already been oversampled by the sampler, '\
                                       'it will be further oversampled by the DA, which may cause errors.')
    else: da_ftns = None
        
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if IS_AUCLOSS: optimizer = config.init_obj('optimizer', module_optim, trainable_params, losses.AUCMLoss(device=device))
    else: optimizer = config.init_obj('optimizer', module_optim, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config.config.keys():
        lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer)
    if lr_scheduler is None: print('lr_scheduler is not set.\n')

    train_kwargs = {
        'model': model,
        'criterion': criterion,
        'metric_ftns': metrics,
        'plottable_metric_ftns': plottable_metric,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'config': config,
        'classes': classes,
        'device': device,
        'data_loader': train_data_loader,
        'valid_data_loader': valid_data_loader,
        'da_ftns': da_ftns
    }
    trainer = Trainer(**train_kwargs) if not IS_FIXED else FixedSpecTrainer(**train_kwargs)

    trainer.train()

    # print the model infomation
    # Option. Use after training because data flows into the model and calculates it
    use_data = next(iter(train_data_loader))[0].to(device)
    input_size = use_data.shape
    logger.info('\nInput_size: {}'.format(input_size))
    model_info = str(summary(model, input_size=input_size, verbose=0))
    logger.info('{}\n'.format(model_info))

    reset_device('cache')
    if config['arch']['visualization']:
        logger.debug('Save the model graph...\n')
        graph_path = config.output_dir / config['arch']['type']
        logger.debug(graph_path)
        make_dot(model(use_data), params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render(graph_path, format='png') 


""" Run """
IS_FIXED, IS_AUCLOSS = init_args()
if IS_AUCLOSS:
    print('auc mode')
    MODEL_BEST_PATH, NEW_AUC_CONFIG_PATH = change_config_for_auclossmode()
    print('auc mode', MODEL_BEST_PATH)
    args = parsing_args(NEW_AUC_CONFIG_PATH)
else: args = parsing_args()
# custom cli options to modify configuration from default values given in json file.
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)

main(config)
