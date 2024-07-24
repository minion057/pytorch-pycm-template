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

""" 
All metrics use the one-vs-rest strategy. 
Metrics that need to be plotted, such as CI and ROC curves, should be written in plottable_metrics.py.
"""



""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('ACC', confusion_obj, positive_class_idx)
def ACC_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('ACC', confusion_obj, classes)

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TPR', confusion_obj, positive_class_idx)
def TPR_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('TPR', confusion_obj, classes)

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TNR', confusion_obj, positive_class_idx)
def TNR_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('TNR', confusion_obj, classes)
    
""" F1 Score """
def F1(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('F1', confusion_obj, positive_class_idx)
def F1_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('F1', confusion_obj, classes)
    
""" PPV, Precision or positive predictive value """
def precision(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('PPV', confusion_obj, positive_class_idx)
def precision_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('PPV', confusion_obj, classes)
    
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
def AUC_OvR_class(confusion_obj:pycmCM, classes=None, method:str='basic'):
    if method not in ['basic', 'roc']: raise ValueError('The method can only be "basic" or "roc".')
    if method.lower() == 'basic': return base_class_metric('AUC', confusion_obj, classes)
    if confusion_obj.prob_vector is None: raise ValueError('No value for prob vector.')
    crv = ROCCurve(actual_vector=confusion_obj.actual_vector, probs=confusion_obj.prob_vector, classes=confusion_obj.classes)
    if classes is None: return crv.area()
    return {classes[class_idx]:v for class_idx, v in enumerate(crv.area().values())}
        