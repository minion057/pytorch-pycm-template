from pycm import ConfusionMatrix as pycmCM
from copy import deepcopy


def base_metric(ftns_name, confusion_obj:pycmCM, positive_class_idx=None, average_type='Macro'):
    use_confusion_obj = deepcopy(confusion_obj)
    if positive_class_idx is None:
        if ftns_name in ['ACC', 'RACC', 'RACCU', 'J', 'CEN', 'MCEN', 'MCC'] and average_type == 'Overall': 
            metric_name = f'Overall_{ftns_name}'
        elif ftns_name in ['ACC', 'PPV', 'NPV', 'TPR', 'TNR', 'FPR', 'FNR', 'F1']:
            if average_type not in ['Macro', 'Micro']: raise TypeError('The metric you set up can be averaged macro or micro.')
            if average_type == 'Micro' and ftns_name == 'ACC': raise TypeError('The micro method is not supported for accuracy.')
            metric_name = f'{ftns_name}_{average_type}'
        else: metric_name = ftns_name
        if metric_name not in dir(confusion_obj): raise ValueError(f'This metric ({ftns_name}) is not supported by the pycm library.')
        score = eval(f'use_confusion_obj.{metric_name}')
    else:
        if ftns_name not in dir(confusion_obj): raise ValueError(f'This metric ({ftns_name}) is not supported by the pycm library.')
        # classes = list(use_confusion_obj.classes)
        score = eval(f'use_confusion_obj.{ftns_name}')
        score = score[list(score.keys())[positive_class_idx]]
    return score if score != 'None' else 0.
    
def base_class_metric(ftns_name, confusion_obj:pycmCM, classes=None, **kwargs):
    use_confusion_obj = deepcopy(confusion_obj)
    if ftns_name not in dir(confusion_obj): raise ValueError(f'This metric ({ftns_name}) is not supported by the pycm library.')
    result = eval(f'use_confusion_obj.{ftns_name}') if not kwargs else eval(f'use_confusion_obj.{ftns_name}(**kwargs)')
    if classes is not None and classes != list(use_confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(result.values())}
    else: return result