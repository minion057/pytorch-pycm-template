from pycm import ConfusionMatrix as pycmCM
from copy import deepcopy

def specificity(confusion:pycmCM, classes, negative_idx:int=None):
    confusion_obj = deepcopy(confusion)
    if negative_idx is None:
        for idx, c in enumerate(classes):
            if 'normal' == c.lower(): negative_idx = idx
    if negative_idx is None: raise('The normal class cannot be identified. Specificity is seeing negatives as normal.')

    if confusion_obj.classes[negative_idx] in confusion_obj.TPR.keys(): 
        spec = confusion_obj.TPR[confusion_obj.classes[negative_idx]]
    else: spec = 0
    return spec if spec != 'None' else 0.
    
def sensitivity(confusion:pycmCM, classes, negative_idx:int=None):
    confusion_obj = deepcopy(confusion)
    if negative_idx is None:
        for idx, c in enumerate(classes):
            if 'normal' == c.lower(): negative_idx = idx
    sens = confusion_obj.TPR
    if negative_idx is None: print('Warring: The normal class cannot be identified. Sensitivity considers normal or a different condition from normal as negative and cancer as positive.')
    elif confusion_obj.classes[negative_idx] in sens.keys(): del sens[confusion_obj.classes[negative_idx]]

    sens = {k:(v if v != 'None' else 0. ) for k, v in sens.items()}
    if len(classes) == 2: return list(sens.values())[0]
    return sens

'''
from sklearn.metrics import recall_score

def specificity_sklearn(actual, pred, negative_idx):
    return recall_score(actual, pred, labels=[negative_idx], average=None)[0]
    
def sensitivity_sklearn(actual, pred, cancer_index):
    return recall_score(actual, pred, labels=[cancer_index], average=None)[0]
'''