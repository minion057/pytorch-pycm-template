from pathlib import Path

CUSTOM_MASSAGE = 'Please enter the required parameters and their values.'

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
                value.mkdir(parents=True, exist_ok=True)
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
