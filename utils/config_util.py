from pathlib import Path
from .util import ensure_dir
from collections import OrderedDict

CUSTOM_MASSAGE = 'Please enter the required parameters and their values.'

def set_common_experiment_name(config:dict|OrderedDict):
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError('The config must be a dict or an OrderedDict.')
    
    # Option 1. Learning rate scheduler
    lr_scheduler = '' if 'lr_scheduler' not in config['trainer'].keys() else f"lr_{config['trainer']['lr_scheduler']['type']}"
    # Option 2. Accumulation steps
    acc_steps = '' if 'accumulation_steps' not in config['trainer'].keys() else f"X{config['trainer']['accumulation_steps']}"
    # Option 3. DA (Data Augmentation) 
    da = '' if 'data_augmentation' not in config.keys() else f"DA_{config['data_augmentation']['type']}"
    # Option 4. Sampler
    # Option 4-1. Samplers that use methods that affect the LOSS calculation.
    sampler_without_dataloader = ''
    if 'data_augmentation' in config.keys():
        if 'prob' in config['data_augmentation']['args'].keys():
            if config['data_augmentation']['args']['prob'] is None:
                sampler_without_dataloader = f'O_DA'
    # Option 4-2. Other samplers. (using DA)
    if 'sampler' in config['data_loader']['args'].keys():
        sampling_type = config['data_loader']['args']['sampler']['args']['sampler_type']
        sampling_type = sampling_type.split('-')[0].split('_')[0][0].upper() # Oversampling -> O, Undersampling -> U
        sampler_with_dataloader = f"{sampling_type}_{config['data_loader']['args']['sampler']['args']['sampler_name']}"
    else: sampler_with_dataloader = ''
    if sampler_without_dataloader != '' and sampler_with_dataloader != '':
        sampling = f"combine_{sampler_with_dataloader}_{sampler_without_dataloader}"
    elif sampler_without_dataloader != '':
        sampling = sampler_without_dataloader
    elif sampler_with_dataloader != '':
        sampling = sampler_with_dataloader
    else: sampling = ''
    
    exper_name = (
        # Required elements in the 1st folder: Name of config
        f"{config['name']}"
        # Required elements in the 2st folder: Name of model and dataloader
        f"/{config['arch']['type']}-{config['data_loader']['type']}"
        # Required elements in the 3st folder: Optimizer and learning rate
        # Optional elements in the 3st folder: Learning rate scheduler that anneals the learning rate                
        f"/{config['optimizer']['type']}-lr_{config['optimizer']['args']['lr']}{lr_scheduler}"
        # Required elements in the 4st folder: Name of loss function
        # Optional elements in the 4st folder: Data augmentation and sampler
        f"/{config['loss']}{'' if da == '' else f'-{da}'}{'' if sampling == ''else f'-{sampling}'}"
        # Required elements in the 5st folder: Batch size and number of epochs
        # Optional elements in the 5st folder: accumulation steps
        f"/{config['data_loader']['args']['batch_size']}batch-{config['trainer']['epochs']}epoch{acc_steps}"
    )
    return exper_name

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
