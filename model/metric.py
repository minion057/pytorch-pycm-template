import traceback
try:
    from .metric_custom import *
except:
    print('Please check if "model/metric_custom.py" exists, and if it does, add its path to sys.path.')
    print('If necessary, please create and write the file. If not necessary, please comment out this section.')
    print(traceback.format_exc())

from pycm import ConfusionMatrix as pycmCM
from pycm import ROCCurve
from base import base_metric, base_class_metric
import numpy as np

""" 
All metrics use the one-vs-rest strategy. 
Metrics that need to be plotted, such as CI and ROC curves, should be written in plottable_metrics.py.
"""



""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, positive_class_idx=None, average_type='Macro'):
    return base_metric('ACC', confusion_obj, positive_class_idx, average_type)
def ACC_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    return base_class_metric('ACC', confusion_obj, classes, positive_class_indices)

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TPR', confusion_obj, positive_class_idx)
def TPR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    return base_class_metric('TPR', confusion_obj, classes, positive_class_indices)

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TNR', confusion_obj, positive_class_idx)
def TNR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    return base_class_metric('TNR', confusion_obj, classes, positive_class_indices)
    
""" F1 Score """
def F1(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('F1', confusion_obj, positive_class_idx)
def F1_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    return base_class_metric('F1', confusion_obj, classes, positive_class_indices)
    
""" PPV, Precision or positive predictive value """
def precision(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('PPV', confusion_obj, positive_class_idx)
def precision_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    return base_class_metric('PPV', confusion_obj, classes, positive_class_indices)
    
""" 
AUC (Area under the ROC curve)
Warring: The AUC calculated using the ROC curve and the AUC calculated using the default function can have different values.
ROC CURVE calculates the threshold in detail. 
AUC is calculated in a simpler way, which may result in a different value than the AUC obtained through ROC CURVE.
"""
def AUC_OvR(confusion_obj:pycmCM, classes=None, positive_class_idx=None, method:str='basic'):
    if positive_class_idx is None: raise ValueError('AUC requires positive_class_idx.')
    if method not in ['basic', 'roc']: raise ValueError('The method can only be "basic" or "roc".')
    if method.lower() == 'basic': return base_metric('AUC', confusion_obj, positive_class_idx)   
    auc = AUC_OvR_class(confusion_obj, classes, method)
    if classes is None: return auc[list(acu.keys)[positive_class_idx]] 
    return auc[classes[positive_class_idx]]
def AUC_OvR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None, method:str='basic'):
    if method not in ['basic', 'roc']: raise ValueError('The method can only be "basic" or "roc".')
    if method.lower() == 'basic': return base_class_metric('AUC', confusion_obj, classes, positive_class_indices)
    if confusion_obj.prob_vector is None: raise ValueError('No value for prob vector.')
    if isinstance(confusion_obj.prob_vector, dict):
        confusion_obj.prob_vector = confusion_obj.prob_vector['all_probs']
    label_classes = list(np.unique(confusion_obj.actual_vector))
    if list(label_classes) != list(confusion_obj.classes): 
        if len(label_classes) == 1: # auc 면적을 구할 수 없는 상태
            if classes is None: return {c:0. for c in confusion_obj.classes}
            return {c:0. for c in classes}
        print(f'경고: 현재 label에 존재하는 class({label_classes})와 classes({confusion_obj.classes})가 일치하지 않습니다.')
    crv = ROCCurve(actual_vector=confusion_obj.actual_vector, probs=confusion_obj.prob_vector, classes=label_classes)
    if classes is None: return crv.area()
    return {classes[class_idx]:v for class_idx, v in enumerate(crv.area().values())}
        