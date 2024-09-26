import traceback
try:
    from .metric_custom import *
except:
    print('Please check if "model/metric_custom.py" exists, and if it does, add its path to sys.path.')
    print('If necessary, please create and write the file. If not necessary, please comment out this section.')
    print(traceback.format_exc())

import numpy as np
from pycm import ConfusionMatrix as pycmCM
from pycm import ROCCurve
from base import BaseMetricFtns

""" 
All metrics use the one-vs-rest strategy. 
Metrics that need to be plotted, such as CI and ROC curves, should be written in plottable_metrics.py.
"""



""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, positive_class_idx=None, average_type='Macro'):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('ACC', positive_class_idx, average_type)
def ACC_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('ACC', positive_class_indices)

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('TPR', positive_class_idx)
def TPR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('TPR', positive_class_indices)

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('TNR', positive_class_idx)
def TNR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('TNR', positive_class_indices)
    
""" F1 Score """
def F1(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('F1', positive_class_idx)
def F1_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('F1', positive_class_indices)
    
""" PPV, Precision or positive predictive value """
def precision(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('PPV', positive_class_idx)
def precision_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('PPV', positive_class_indices)
    
""" 
AUC (Area under the ROC curve)
Warring: The AUC calculated using the ROC curve and the AUC calculated using the default function can have different values.
ROC CURVE calculates the threshold in detail. 
AUC is calculated in a simpler way, which may result in a different value than the AUC obtained through ROC CURVE.
"""
def AUC_OvR(confusion_obj:pycmCM, classes=None, positive_class_idx=None, method:str='basic'):
    if method not in ['basic', 'roc']: raise ValueError('The method can only be "basic" or "roc".')
    if positive_class_idx is None: raise ValueError('AUC requires positive_class_idx.')
    return list(AUC_OvR_class(confusion_obj, classes, [positive_class_idx], method).values())[0]
def AUC_OvR_class(confusion_obj:pycmCM, classes=None, positive_class_indices=None, method:str='basic'):
    if method not in ['basic', 'roc']: raise ValueError('The method can only be "basic" or "roc".')
    
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    ftns_name = 'AUC'
    
    # basic version
    if method.lower() == 'basic': return ftns.multi_class_metric(ftns_name, positive_class_indices)
    # roc version
    if confusion_obj.prob_vector is None: raise ValueError('No value for prob vector.')
    if isinstance(confusion_obj.prob_vector, dict):
        confusion_obj.prob_vector = confusion_obj.prob_vector['all_probs']
    target_classes = list(np.unique(confusion_obj.actual_vector))
    if list(target_classes) != list(confusion_obj.classes): 
        if len(target_classes) == 1: # AUC area not available
            if classes is None: return {c:0. for c in confusion_obj.classes}
            return {c:0. for c in classes}
        print(f"Warning: Classes in current label ({target_classes}) do not match classes in confusion object ({confusion_obj.classes}). Please ensure that the classes are consistent.")
    crv = ROCCurve(actual_vector=confusion_obj.actual_vector, probs=confusion_obj.prob_vector, classes=target_classes)
    return ftns._format_metric_for_multi_score(ftns_name, crv.area(), positive_class_indices)