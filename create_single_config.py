from utils import CUSTOM_MASSAGE, yes_or_no, monitor_value
from utils import type_int, type_float, type_attr, type_path, type_metrics
from utils import write_json

from collections import OrderedDict
from pathlib import Path
import numpy as np
import model.model as module_arch
import data_loader.__init__ as module_data
import model.optim as module_optim
import torch.optim.lr_scheduler as module_lr_s
import model.loss as module_loss
import model.metric_curve_plot as module_curve_metric
import model.metric as module_metric
import data_loader.data_augmentation as module_DA
import data_loader.data_sampling as module_sampling

def main():
    config = OrderedDict()

    # basic information
    print('To easily distinguish config files, the additional information will be appended after the name you entered.')
    print('Now setting: {config/{name}/{optim}-lr_{lr}-{lr_scheduler}/{batch_size}X{accumulation_steps}-{epoch}epoch-{loss}-{DA}-{Sampling}-model.json}')
    config['name']=input("Please enter the folder name to save the outputs. : ")
    config['n_gpu']=type_int('\nPlease enter the number of GPUs to use. : ')

    # model information
    config['arch']=OrderedDict()
    print('\nPlease refer to the \'model/model.py\' document for available models.')
    config['arch']['type']=type_attr(module_arch, 'Please enter the model to use. : ')
    config['arch']['args']=OrderedDict()
    config['arch']['args']['num_classes']=type_int('\nPlease enter the number of classes in the dataset. : ')
    config['arch']['args']['custom']=CUSTOM_MASSAGE
    config['arch']['visualization'] = yes_or_no('Would you like to visualize the model structure as a graph? ')
    
    # dataloader information
    config['data_loader']=OrderedDict()
    print('\nPlease refer to the \'data_loader/__init__.py\' document for available dataloaders.')
    config['data_loader']['type']=type_attr(module_data, 'Please enter the name of the data loader to use. : ')
    config['data_loader']['args']=OrderedDict()
    config['data_loader']['args']['batch_size']=type_int('\nPlease enter the batch size. : ')
    config['data_loader']['args']['custom']=CUSTOM_MASSAGE
    
    # data augmentation information
    print('\nPlease refer to the \'data_loader/data_augmentation.py\' document for available funtion.')
    if yes_or_no('Will you use a data augmentation techniques? '):
        config['data_augmentation']=OrderedDict()
        config['data_augmentation']['type']=type_attr(module_DA, "Please enter the name of the data augmentation technique you would like to use. : ")
        # input("Please enter the name of the data augmentation technique you would like to use. : ")
        config['data_augmentation']['args']=OrderedDict()
        config['data_augmentation']['args']['custom']=CUSTOM_MASSAGE
        config['data_augmentation']['hook_args']=OrderedDict()
        config['data_augmentation']['hook_args']['layer_idx']=type_int('Please enter the index of the layer you would like to apply the hook to. : ')
        config['data_augmentation']['hook_args']['pre']=yes_or_no('Before executing the forward operation, will you proceed with the hook? ')
    
    # data sampling information
    print('\nPlease refer to the \'data_loader/data_sampling.py\' document for available funtion.')
    if yes_or_no('Will you use a data sampling techniques? '):
        config['data_sampling']=OrderedDict()
        config['data_sampling']['type']=str(input("Please enter the type of the data sampling technique you would like to use. (Up or Down) : ")).lower()
        config['data_sampling']['name']=type_attr(module_sampling, "Please enter the name of the data sampling technique you would like to use. : ")
        # input("Please enter the name of the data sampling technique you would like to use. : ")
        # config['data_sampling']['args']=OrderedDict()
        # config['data_sampling']['args']['custom']=CUSTOM_MASSAGE
    
    # optimizer information
    config['optimizer']=OrderedDict()
    print('\nPlease refer to the \'model/optim.py\' document for available optimizers.')
    config['optimizer']['type']=type_attr(module_optim, 'Please enter the name of the optimizer to use. : ')
    config['optimizer']['args']=OrderedDict()
    config['optimizer']['args']['lr']=type_float('\nPlease enter the learning rate. : ')
    config['optimizer']['args']['custom']=CUSTOM_MASSAGE

    # learning rate scheduler information
    if yes_or_no('Will you use a learning rate scheduler? '):
        config['lr_scheduler']=OrderedDict()
        config['lr_scheduler']['type']=type_attr(module_lr_s, '\nPlease enter the learning rate scheduler to use. : ')
        config['lr_scheduler']['args']=OrderedDict()
        config['lr_scheduler']['args']['step_size']=type_int('\nPlease enter the step size. : ')
        config['lr_scheduler']['args']['custom']=CUSTOM_MASSAGE

    # loss information
    print('\nPlease refer to the \'model/loss.py\' document for available loss funtion.')
    config['loss']=type_attr(module_loss, 'Please enter the loss function to use. : ')
    
    # metrics information
    config['metrics']=type_metrics(module_metric)
    if yes_or_no('Will you be using curve metric functions such as ROC? '):
        config['curve_metrics']=type_metrics(module_curve_metric)
        
    # trainer information
    config['trainer']=OrderedDict()
    config['trainer']['epochs']=type_int('\nPlease indicate the desired number of epochs for training. : ')
    if yes_or_no('Do you want to set a accumulation_steps? '):
        config['trainer']['accumulation_steps']=type_int('\nPlease indicate the desired number of accumulation_steps for training. : ')

    save_path=f"saved/"
    print(f'\nIf you don\'t specify a save path for the output files, ',\
          f'they will be automatically saved in the \'{save_path}\' folder within the current execution directory.')
    if yes_or_no('Do you want to set the file save path? '): save_path=type_path('\nPlease enter the path. : ')
    config['trainer']['save_dir']=save_path
    config['trainer']['save_period']=type_int('\nPlease enter the frequency to save checkpoints during training. (Recommendation: 1) : ', range=list(np.arange(1, config['trainer']['epochs']+1)))
    config['trainer']['verbosity']=type_int('\nPlease enter the logger output level. (0: quiet, 1: per epoch, 2: full) : ', range=[0,1,2])

    if yes_or_no('Do you want to monitor a specific value to find the best model? '):
        value=monitor_value(config['metrics'])
        monitor='min' if yes_or_no('Do you want to monitor for the minimum value?\nIf you answer \'no\', it will be set as a maximum value. : ') else 'max'
        config['trainer']['monitor']=f'{monitor} {value}'
        if yes_or_no('Will you be using early stopping? '):
            config['trainer']['early_stop']=type_int('\nPlease enter the number of early stopping iterations. : ', range=list(np.arange(1, config['trainer']['epochs']+1)))
    
    config['trainer']['tensorboard']=False
    config['trainer']['tensorboard_projector']=OrderedDict()
    config['trainer']['tensorboard_projector']['train']=yes_or_no('Would you like to visualize the training dataset in the TensorBoard projector? ')
    config['trainer']['tensorboard_projector']['valid']=yes_or_no('Would you like to visualize the validation dataset in the TensorBoard projector? ')
    config['trainer']['tensorboard_pred_plot']=yes_or_no('Would you like to log prediction examples on TensorBoard? ')
    if config['trainer']['tensorboard_projector']['train'] or config['trainer']['tensorboard_projector']['valid'] or config['trainer']['tensorboard_pred_plot']: config['trainer']['tensorboard']=True
    config['trainer']['save_performance_plot']=yes_or_no('Do you want to save performance plots for each epoch? ')
    
    # tester information
    config['tester']=OrderedDict()
    config['tester']['tensorboard_projector']=yes_or_no('Would you like to visualize the test dataset in the TensorBoard projector? ')
    
    # Saving the file.
    save_path=f"config/{config['name']}/{config['optimizer']['type']}-lr_{config['optimizer']['args']['lr']}"
    if 'lr_scheduler' in config.keys(): save_path+=f"-{config['lr_scheduler']['type']}"
    print(f'\nIf you don\'t set a file save path, it will be saved in the {save_path} folder in the current execution directory.')
    if yes_or_no('Do you want to set the file save path? '): save_path=type_path('\nPlease enter the file path. : ')
    else:
        try: Path(save_path).mkdir(parents=True, exist_ok=True)
        except Exception as e: print(e)
    
    # config_file_name=f"{config['arch']['type']}.json"
    acc_steps = '' if 'accumulation_steps' not in config['trainer'].keys() else f"X{config['trainer']['accumulation_steps']}"
    sampling = '' if 'data_sampling' not in config.keys() else f"-{config['data_sampling']['type']}_{config['data_sampling']['name']}"
    da = '' if 'data_augmentation' not in config.keys() else f"-{config['data_augmentation']['type']}"
    config_file_name =f"{config['data_loader']['args']['batch_size']}batch{acc_steps}-{config['trainer']['epochs']}epoch"
    config_file_name+=f"-{config['loss']}{da}{sampling}-{config['arch']['type']}.json"
    config_file_path=Path(save_path) / config_file_name
    write_json(config, config_file_path)

    # Done
    print('\nAll configurations have been written, and the save path is as follows.')
    print(config_file_path)
    print('\nPlease check the parts that you need to fill out.')

main()