import numpy as np
from pycm import ConfusionMatrix as pycmCM
from copy import deepcopy

class BaseMetricFtns:
    def __init__(self, confusion_obj:pycmCM, classes=None):
        self.confusion_obj = deepcopy(confusion_obj)
        self.classes = classes
        self.confusion_classes = deepcopy(confusion_obj.classes)
        self.confusion_classes_is_index_list = False
        self._check_class_match()
            
    def _check_class_match(self):
        for confusion_class_item in self.confusion_classes: 
            if isinstance(confusion_class_item, int): 
                if self.classes is None:
                    if confusion_class_item not in list(range(len(self.classes))):
                        raise ValueError(f"Expected an integer value between 0 and {len(self.classes)-1}, but received {confusion_class_item}.")
                else:
                    if confusion_class_item not in self.classes:
                        raise TypeError(f"Classes set in configuration do not match classes present in score.")
                self.confusion_classes_is_index_list = True
            else:
                if self.classes is not None and confusion_class_item not in self.classes:
                    raise TypeError(f"Classes set in configuration do not match classes present in score.\n"+
                                    f"Now, classes: {self.classes}, confusion_classes: {self.confusion_classes}.")
    
    def _check_metric_support(self, ftns_name):
        if ftns_name not in dir(self.confusion_obj): 
            raise ValueError(f'This metric ({ftns_name}) is not supported by the pycm library.')
    
    def _format_metric_for_single_score(self, ftns_name, average_type='Macro') -> float:
        if ftns_name in ['ACC', 'RACC', 'RACCU', 'J', 'CEN', 'MCEN', 'MCC'] and average_type == 'Overall': 
            metric_name = f'Overall_{ftns_name}'
        elif ftns_name in ['ACC', 'PPV', 'NPV', 'TPR', 'TNR', 'FPR', 'FNR', 'F1']:
            if average_type not in ['Macro', 'Micro']: raise TypeError('The metric you set up can be averaged macro or micro.')
            if average_type == 'Micro' and ftns_name == 'ACC': raise TypeError('The micro method is not supported for accuracy.')
            metric_name = f'{ftns_name}_{average_type}'
        else: metric_name = ftns_name
        self._check_metric_support(ftns_name)
        score = eval(f'self.confusion_obj.{metric_name}')
        if isinstance(score, dict):
            raise ValueError("Metric requires 'positive_class_idx' but it is not available. "+
                            "Please provide a valid 'positive_class_idx' to proceed. "+
                            "Or use the class version of the metrics function.")
        return score if score != 'None' else 0.
        
    def _format_metric_for_multi_score(self, ftns_name, score, positive_class_indices=None) -> dict:
        if self.classes is not None:
            if self.confusion_classes_is_index_list:
                ori_score = {self.classes[class_idx]:v for class_idx, v in score.items()} 
            else: ori_score = {class_item:v for class_item, v in score.items()}
        ori_score = {class_item:v if v != 'None' else 0. for class_item, v in ori_score.items()}
        if positive_class_indices is not None:
            positive_classes = np.array(self.classes)[positive_class_indices]
            score = deepcopy(ori_score)
            for class_item in ori_score.keys():
                if class_item not in positive_classes: del score[class_item]
        return score
    
    def single_class_metric(self, ftns_name:str, positive_class_idx=None, average_type='Macro') -> float:
        # 1. Returning only a single metric score.
        if positive_class_idx is None: return self._format_metric_for_single_score(ftns_name, average_type)
        # 2. Returns the metric score for all classes.
        if positive_class_idx is None: raise ValueError('Single score requires positive_class_idx.')
        return list(self.multi_class_metric(ftns_name, positive_class_indices=[positive_class_idx]).values())[0]
    
    def multi_class_metric(self, ftns_name:str, positive_class_indices=None, **kwargs) -> dict:
        # Returns the metric score for all classes.
        self._check_metric_support(ftns_name)
        score = eval(f'self.confusion_obj.{ftns_name}') if not kwargs else eval(f'self.confusion_obj.{ftns_name}(**kwargs)')
        return self._format_metric_for_multi_score(ftns_name, score, positive_class_indices)