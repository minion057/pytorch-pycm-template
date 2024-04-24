import traceback
try:
    from .metric_custiom import *
except:
    print('Please check if "model/metric_custiom.py" exists, and if it does, add its path to sys.path.')
    print('If necessary, please create and write the file. If not necessary, please comment out this section.')
    print(traceback.format_exc())

from pycm import ConfusionMatrix as pycmCM


""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    if specific_class_idx is None: return confusion_obj.Overall_ACC
    else:
        classes = list(confusion_obj.classes)
        score = confusion_obj.ACC[classes[specific_class_idx]]
        return score if score != 'None' else 0.
def ACC_class(confusion_obj:pycmCM, classes=None):
    if classes is not None and classes != list(confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.ACC.values())}
    else: return confusion_obj.ACC

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    if specific_class_idx is None: # Returns the macro-average of True Positive Rate (TPR) across all classes.
        return confusion_obj.TPR_Macro if confusion_obj.TPR_Macro != 'None' else 0.
    else: # Returns the True Positive Rate (TPR) when a specific class is considered positive.
        classes = list(confusion_obj.classes)
        score = confusion_obj.TPR[classes[specific_class_idx]]
        return score if score != 'None' else 0.

def TPR_class(confusion_obj:pycmCM, classes=None):
    if classes is not None and list(classes) != list(confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.TPR.values())}
    else: return confusion_obj.TPR

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    if specific_class_idx is None: # Returns the macro-average of TNR across all classes.
        return confusion_obj.TNR_Macro if confusion_obj.TNR_Macro != 'None' else 0.
    else: # Returns the TNR when a specific class is considered positive.
        classes = list(confusion_obj.classes)
        score = confusion_obj.TNR[classes[specific_class_idx]]
        return score if score != 'None' else 0.
def TNR_class(confusion_obj:pycmCM, classes=None):
    if classes is not None and list(classes) != list(confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.TNR.values())}
    else: return confusion_obj.TNR