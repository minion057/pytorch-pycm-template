import numpy as np
from pycm import ConfusionMatrix as pycmCM
from base import BaseMetricFtns
from utils import check_and_import_library, close_all_plots

""" 
All metrics use the one-vs-rest strategy. 
Metrics that need to be plotted, such as CI and ROC curves, should be written in plottable_metrics.py.
"""



""" Accuracy """
def CBA(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Class balance accuracy
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('CBA', positive_class_idx)
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
def BACC(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('TPR', positive_class_idx, 'Macro')

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
def AUC(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.single_class_metric('AUC', positive_class_idx)
def AUC_class(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    # Instantiate objects and verify class correctness
    ftns = BaseMetricFtns(confusion_obj=confusion_obj, classes=classes)
    return ftns.multi_class_metric('AUC', positive_class_indices)

def AUC_OvR(confusion_obj:pycmCM, classes, positive_class_idx=None):
    if positive_class_idx is None: raise ValueError('AUC requires positive_class_idx.')
    return list(AUC_OvR_class(confusion_obj, classes, [positive_class_idx]).values())[0]
def AUC_OvR_class(confusion_obj:pycmCM, classes, positive_class_indices=None):    
    if classes is None: raise ValueError('CLASSES is required to be entered in order to calculate with the ROC version.')
    ftns_name, metrics_module = 'ROC_OvR', check_and_import_library('model.plottable_metrics')
    if ftns_name not in dir(metrics_module): raise ValueError(f'Warring: {ftns_name} is not in the model.metric library.')
    use_ftns = getattr(metrics_module, ftns_name)
    
    target_classes = list(np.unique(confusion_obj.actual_vector))
    if len(classes) != len(target_classes): return {c:-1. for c in classes} # AUC area not available
    if not all(class_item in list(target_classes) for class_item in list(classes)):
        print(f"Warning: Classes in current label ({target_classes}) do not match classes in confusion object ({classes})."
              +"Please ensure that the classes are consistent.")
    
    if confusion_obj.prob_vector is None: raise ValueError('No value for prob vector.')
    if isinstance(confusion_obj.prob_vector, dict):
        confusion_obj.prob_vector = confusion_obj.prob_vector['all_probs']
    roc_dict, _ = use_ftns(labels=confusion_obj.actual_vector, probs=confusion_obj.prob_vector, 
                           classes=classes, positive_class_indices=list(range(len(classes))), return_result=True)
    close_all_plots()
    real_score = {classes[pos]:auc for (pos, neg), auc in zip(roc_dict['pos_neg_idx'], roc_dict['auc'])}
    score = {}
    for class_item in classes:
        if class_item not in real_score.keys(): 
            score[class_item] = -1.
        else: score[class_item] = real_score[class_item]
    if positive_class_indices is not None: 
        positive_classes = np.array(classes)[positive_class_indices]
        for class_item in list(score.keys()):
            if class_item not in positive_classes: del score[class_item]
    return score

def AUC_OvO(confusion_obj:pycmCM, classes, positive_class_idx=None):
    if positive_class_idx is None: raise ValueError('AUC requires positive_class_idx.')
    return list(AUC_OvO_class(confusion_obj, classes, [positive_class_idx]).values())[0]
def AUC_OvO_class(confusion_obj:pycmCM, classes, positive_class_indices=None):    
    if classes is None: raise ValueError('CLASSES is required to be entered in order to calculate with the ROC version.')
    ftns_name, metrics_module = 'ROC_OvO', check_and_import_library('model.plottable_metrics')
    if ftns_name not in dir(metrics_module): raise ValueError(f'Warring: {ftns_name} is not in the model.metric library.')
    use_ftns = getattr(metrics_module, ftns_name)
    
    target_classes = list(np.unique(confusion_obj.actual_vector))
    if len(target_classes) == 1: return {c:-1. for c in classes} # AUC area not available
    if not all(class_item in list(target_classes) for class_item in list(classes)):
        print(f"Warning: Classes in current label ({target_classes}) do not match classes in confusion object ({classes})."
              +"Please ensure that the classes are consistent.")
    
    if confusion_obj.prob_vector is None: raise ValueError('No value for prob vector.')
    if isinstance(confusion_obj.prob_vector, dict):
        confusion_obj.prob_vector = confusion_obj.prob_vector['all_probs']
        
    roc_dict, _ = use_ftns(labels=confusion_obj.actual_vector, probs=confusion_obj.prob_vector, classes=classes, return_result=True, 
                           positive_class_indices=list(range(len(classes))), negative_class_indices=list(range(len(classes))))
    close_all_plots()
    real_score = {f'{classes[pos]} vs {classes[neg]}':auc for (pos, neg), auc in zip(roc_dict['pos_neg_idx'], roc_dict['auc'])}
    score = {}
    for pos_class in classes:
        for neg_class in classes:
            if pos_class == neg_class: continue
            socre_key = f'{pos_class} vs {neg_class}'
            if socre_key not in real_score.keys(): score[socre_key] = -1.
            else: score[socre_key] = real_score[socre_key]
    if positive_class_indices is not None: 
        positive_classes = np.array(classes)[positive_class_indices]
        for class_item in list(score.keys()):
            pos_class = class_item.split(' vs ')[0]
            if pos_class not in positive_classes: del score[class_item]
    return score