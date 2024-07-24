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
        score = eval(f'use_confusion_obj.{ftns_name}')
        score = score[list(score.keys())[positive_class_idx]]
    return score if score != 'None' else 0.
    
def base_class_metric(ftns_name, confusion_obj:pycmCM, classes=None, **kwargs):
    use_confusion_obj = deepcopy(confusion_obj)
    if ftns_name not in dir(confusion_obj): raise ValueError(f'This metric ({ftns_name}) is not supported by the pycm library.')
    score = eval(f'use_confusion_obj.{ftns_name}') if not kwargs else eval(f'use_confusion_obj.{ftns_name}(**kwargs)')
    score_classes = list(score.keys()) 
    if classes is not None:
        if len(classes) != len(score_classes): raise ValueError('The number of set classes and the number of classes in the confusion matrix do not match.')
        if list(classes) != list(score_classes): return {classes[class_idx]:v for class_idx, v in enumerate(score.values())}
    else: return score