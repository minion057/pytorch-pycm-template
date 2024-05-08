from pycm import ConfusionMatrix as pycmCM

def base_metric(ftns_name, confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    if positive_class_idx is None:
        metric_name = f'{ftns_name}_Macro' if ftns_name != 'ACC' else 'Overall_ACC'
        return eval(f'confusion_obj.{metric_name}')
    else:
        classes = list(confusion_obj.classes)
        score = eval(f'confusion_obj.{ftns_name}[classes[positive_class_idx]]')
        return score if score != 'None' else 0.
    
def base_class_metric(ftns_name, confusion_obj:pycmCM, classes=None):
    if classes is not None and classes != list(confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(eval(f'confusion_obj.{ftns_name}.values()'))}
    else: return eval(f'confusion_obj.{ftns_name}')