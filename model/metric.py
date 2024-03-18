from pycm import ConfusionMatrix as pycmCM


""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    return confusion_obj.Overall_ACC if specific_class_idx is None else confusion_obj.ACC[specific_class_idx]
def ACC_class(confusion_obj:pycmCM, classes=None):
    # return confusion_obj.ACC
    return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.ACC.values())}

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    # 전체 클래스의 TPR 평균 반환
    if specific_class_idx is None: return confusion_obj.TPR_Macro if confusion_obj.TPR_Macro != 'None' else 0.
    # 특정(i.g,암) 클래스의 TPR 평균 반환
    return confusion_obj.TPR[specific_class_idx]
def TPR_class(confusion_obj:pycmCM, classes=None):
    # return confusion_obj.TPR
    return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.TPR.values())}

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, specific_class_idx=None):
    # 전체 클래스의 TNR 평균 반환
    if specific_class_idx is None: return confusion_obj.TNR_Macro if confusion_obj.TNR_Macro != 'None' else 0.
    # 특정(i.g,정상) 클래스의 TNR 평균 
    return confusion_obj.TNR[specific_class_idx]
def TNR_class(confusion_obj:pycmCM, classes=None):
    # return confusion_obj.TNR
    return {classes[class_idx]:score for class_idx, score in enumerate(confusion_obj.TNR.values())}