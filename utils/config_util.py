from pathlib import Path
from .util import ensure_dir
from collections import OrderedDict

CUSTOM_MASSAGE = 'Please enter the required parameters and their values.'

def set_common_experiment_name(config:dict|OrderedDict, return_type:type[str|list|dict]=str):
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError('The config must be a dict or an OrderedDict.')
    if return_type not in [str, list, dict]:
        raise ValueError('The return_type must be either "str" or "list" or "dict".')
    
    exper_dict = OrderedDict({
        'config_name': config['name'],
        'model': config['arch']['type'],
        'dataloader': config['data_loader']['type'],
        'optimizer': config['optimizer']['type'],
        'lr': config['optimizer']['args']['lr'],
        'lr_scheduler': '',
        'loss': config['loss'],
        'da': '',
        'sampler_type': '',
        'sampler': '',
        'batch_size': config['data_loader']['args']['batch_size'],
        'accum_steps': '',
        'max_epoch': config['trainer']['epochs']
    })
    
    # Option 1. Learning rate scheduler
    exper_dict['lr_scheduler'] = '' if 'lr_scheduler' not in config.keys() else config['lr_scheduler']['type']
    # Option 2. Data Augmentation
    exper_dict['da'] = '' if 'data_augmentation' not in config.keys() else config['data_augmentation']['type']
    # Option 3. Sampling
    # Sampler 1. Samplers that use methods that affect the LOSS calculation.
    sampler_without_dataloader, sampler_with_dataloader = {'type':'', 'name':''}, {'type':'', 'name':''}
    if 'data_augmentation' in config.keys():
        if 'prob' in config['data_augmentation']['args'].keys():
            if config['data_augmentation']['args']['prob'] is None:
                sampler_without_dataloader['type'], sampler_without_dataloader['name'] = 'O', 'DA'
    # Sampler 2. Other samplers. (using DA)
    if 'sampler' in config['data_loader']['args'].keys():
        # Oversampling -> O, Undersampling -> U
        sampler_with_dataloader['type'] = config['data_loader']['args']['sampler']['args']['sampler_type'].split('-')[0].split('_')[0][0].upper() 
        sampler_with_dataloader['name'] = config['data_loader']['args']['sampler']['args']['sampler_name']

    if sampler_without_dataloader['type'] != '' and sampler_with_dataloader['type'] != '':
        exper_dict['sampler_type'] = 'combine'
        exper_dict['sampler'] = (
            f"{sampler_without_dataloader['type']}_{sampler_without_dataloader['name']}"
            f"_{sampler_with_dataloader['type']}_{sampler_with_dataloader['name']}"
        )
    elif sampler_without_dataloader['type'] != '':
        exper_dict['sampler_type'], exper_dict['sampler'] = sampler_without_dataloader['type'], sampler_without_dataloader['name']
    elif sampler_with_dataloader['type'] != '':
        exper_dict['sampler_type'], exper_dict['sampler'] = sampler_with_dataloader['type'], sampler_with_dataloader['name']
    # Option 4. Accumulation steps
    exper_dict['accum_steps'] = '' if 'accumulation_steps' not in config['trainer'].keys() else config['trainer']['accumulation_steps']
    
    # Option for duplicate naming
    npz_file_name = ''
    if 'fold' in config['data_loader']['args']['dataset_path']:
        npz_file_name = '/' + config['data_loader']['args']['dataset_path'].split('/')[-1].split('.')[0]
        
    # return config information
    if return_type is str:
        lr_scheduler, da, accum_steps = exper_dict['lr_scheduler'], exper_dict['da'], exper_dict['accum_steps']
        sampler_type, sampler = exper_dict['sampler_type'], exper_dict['sampler']
        exper_name = (
            # Required elements in the 1st folder: Name of config
            f"{exper_dict['config_name']}"
            # Required elements in the 2st folder: Name of model and dataloader
            f"/{exper_dict['model']}-{exper_dict['dataloader']}"            
            # Required elements in the 3st folder: Optimizer and learning rate
            # Optional elements in the 3st folder: Learning rate scheduler that anneals the learning rate         
            f"/{exper_dict['optimizer']}-lr_{exper_dict['lr']}{'' if lr_scheduler == '' else f'-{lr_scheduler}'}"
            # Required elements in the 4st folder: Name of loss function
            # Optional elements in the 4st folder: Data augmentation and sampler
            f"/{exper_dict['loss']}{'' if da == '' else f'-DA_{da}'}{'' if sampler_type == '' else f'-{sampler_type}_{sampler}'}"
            # Required elements in the 5st folder: Batch size and number of epochs
            # Optional elements in the 5st folder: accumulation steps
            f"/{exper_dict['batch_size']}batch{'' if accum_steps == '' else f'X{accum_steps}'}-{exper_dict['max_epoch']}epoch{npz_file_name}"
        )
        return exper_name
    elif return_type is list:
        return list(exper_dict.values())
    return exper_dict

def shutdown_warning(cnt:int, MAX_INPUT_CNT:int=5):
    if cnt == MAX_INPUT_CNT: raise EOFError('You have made a total of 5 incorrect attempts. Forced termination.')
    print(f'The current number of incorrect attempts is {cnt}.')
    print(f'The program will terminate if there are a total of {MAX_INPUT_CNT} incorrect attempts.\n')

def type_int(message:str, cnt:int=0, range:list=None):
    value, correct = input(message), True
    try: value = int(value)
    except:
        correct = False
        print(f'Please enter only numbers. Now input value is {value}')
    if correct and range is not None:
        if value not in range:
            correct = False
            print(f'Please enter only numbers within the range.')
            print(f'Range: {range}')
            print(f'Now input value is {value}')
    if correct: return value
    else:
        cnt += 1
        shutdown_warning(cnt)
        return type_int(message, cnt, range)

def type_float(message:str, cnt:int=0, range:list=None):
    value, correct = input(message), True
    try: value = float(value)
    except:
        correct = False
        print(f'Please enter only numbers. Now input value is {value}')
    if correct and range is not None:
        if value in range:
            correct = False
            print(f'Please enter only numbers within the range.')
            print(f'Range: {range}')
            print(f'Now input value is {value}')
    if correct: return value
    else:
        cnt += 1
        shutdown_warning(cnt)
        return type_int(message, cnt, range)

def type_attr(attr, message:str, cnt:int=0):
    value, correct = input(message), True
    try: getattr(attr, value)
    except Exception as e:
        correct = False
        print(e)
    if correct: return value
    else:
        cnt += 1
        shutdown_warning(cnt)
        return type_attr(attr, message, cnt)

def type_path(message:str, valuecnt:int=0, file:bool=False):
    value, correct = Path(input(message)), True
    if file:
        if not value.is_file(): correct = False
    else:
        try:
            if not value.is_dir():
                ensure_dir(value, True)
                print('Path creation completed.')
        except Exception as e:
            correct = False
            print(e)
    if correct: return str(value)
    else:
        cnt += 1
        shutdown_warning(cnt)
        return type_path(message, cnt, file)

def type_metrics(attr):    
    metrics=dict()
    while True:
        m = type_attr(attr, '\nPlease enter only one metric function to use. : ')
        if yes_or_no('Would you like to calculate this metric only for a specific class (positive)? '):
            m_idx = input('Please input a specific class. : ')
            if m_idx.isnumeric(): m_idx = int(m_idx)
        else: m_idx = None
        metrics[m] = m_idx
        if not yes_or_no('Do you have any additional metric functions to use? '): break
    return metrics

def yes_or_no(message:str, cnt:int=0):
    print('\nPlease answer the following questions with \'yes(y)\' or \'no(n)\'.')
    value = input(message).lower()
    if value in ['yes', 'y']:
        return True
    elif value in ['no', 'n']:
        return False
    else:
        cnt += 1
        shutdown_warning(cnt)
        return yes_or_no(message, cnt)

def monitor_value(metrics:list, cnt:int=0):
    all_metrics=['loss', 'val_loss']
    all_metrics.extend(metrics)
    all_metrics.extend([f'val_{m}' for m in metrics])
    print(f'Currently estimable value: {all_metrics}')
    value = input('\nPlease enter only one value from the above. ')
    if value in all_metrics: return value
    else:
        cnt += 1
        shutdown_warning(cnt)
        return monitor_value(metrics, cnt)
