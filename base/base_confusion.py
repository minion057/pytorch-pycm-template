import os
import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay
from pycm import ConfusionMatrix as pycmCM
from pycm.pycm_util import threshold_func
import model.metric as module_metric
import model.plottable_metrics  as module_plottable_metric
import matplotlib.pyplot as plt

class ConfusionTracker:
    def __init__(self, *keys, classes, writer=None):
        self.writer = writer
        self.classes = classes        
        self._data = pd.DataFrame(index=keys, columns=['actual', 'predict', 'probability', 'confusion'])
        self.reset()

    def reset(self):
        for key in self._data.index.values:
            self._data['actual'][key], self._data['predict'][key], self._data['probability'][key] = [], [], []
            self._data['confusion'][key] = None #[np.zeros((len(self.classes),), dtype=int).tolist() for _ in range(len(self.classes))]

    def update(self, key, value:dict, set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):
        required_keys = ['actual', 'predict']
        if not all(k in list(value.keys()) for k in required_keys):
            # if 'actual' not in list(value.keys()) or 'predict' not in list(value.keys()):
            raise ValueError(f'Correct answer (actual), predicted value (predict) and option value (probability) are required to update ConfusionTracker.\nNow Value {list(value.keys())}.')
        self._data.actual[key].extend(value['actual'])
        self._data.predict[key].extend(value['predict'])
        if 'probability' in value.keys(): self._data.probability[key].extend(value['probability'])
        
        # A basic confusion matrix is generated based on the class with the highest probability.
        confusion_obj = pycmCM(actual_vector=self._data.actual[key], predict_vector=self._data.predict[key])
        self._data.confusion[key] = confusion_obj.to_array().tolist()

        if img_update or set_title is not None or img_save_dir_path is not None:
            # Perform only when all classes of data are present
            if len(self.classes) != len(np.unique(np.array(self._data.confusion[key]), return_counts=True)[0]): return            
            confusion_plt = self.createConfusionMatrix(key)
            confusion_plt.ax_.set_title(set_title if set_title is not None else f'Confusion matrix - {key}')
        
        if self.writer is not None and img_update:
            self.writer.add_figure('ConfusionMatrix', confusion_plt.figure_)
        if img_save_dir_path is not None:
            confusion_plt.figure_.savefig(Path(img_save_dir_path) / f'ConfusionMatrix{key}.png', dpi=300, bbox_inches='tight')

    def get_actual_vector(self, key):
        return list(self._data.actual[key])
    def get_prediction_vector(self, key):
        return list(self._data.predict[key])
    def get_probability_vector(self, key):
        return list(self._data.probability[key])
    def get_confusion_matrix(self, key):
        return dict(self._data.confusion[key])
    def get_confusion_obj(self, key):
        return pycmCM(actual_vector=self.get_actual_vector(key), predict_vector=self.get_prediction_vector(key))
    def result(self):
        return dict(self._data.confusion)

    def createConfusionMatrix(self, key): 
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(self._data.confusion[key]), display_labels=np.array(self.classes))
        confusion_plt = disp.plot(cmap=plt.cm.binary)
        return confusion_plt
    
class FixedSpecConfusionTracker:
    """ The current metric uses the one-vs-one strategy. """
    def __init__(self, classes, goal_score:list, negative_class_indices:[int, list, np.ndarray], goal_digit:int=2, writer=None):
        self.writer = writer
        self.classes = classes
        
        self.fixed_metrics_ftns = getattr(module_metric, 'specificity')
        self.refer_metrics_ftns = getattr(module_metric, 'sensitivity')
        self.negative_class_indices = {class_idx:class_name for class_idx, class_name in enumerate(self.classes) if class_idx == self.negative_class_idx}
        self.positive_class_indices = {class_idx:class_name for class_idx, class_name in enumerate(self.classes) if class_idx != self.negative_class_idx}
        
        self.goal_digit = goal_digit
        self.goal_score = goal_score
        self.index = [[], [], []]
        for goal in self.goal_score:
            for pos_class_name in self.positive_class_indices.values():
                for neg_class_name in self.negative_class_indices.values():
                    self.index[0].append(goal)
                    self.index[1].append(pos_class_name)
                    self.index[2].append(neg_class_name)
        self._data = pd.DataFrame(index=self.index, columns=['confusion', 'auc', 'fixed_score', 'refer_score', 'tag'])
        self.index = self._data.index.values
        self.reset()
    
    def reset(self):
        self.actual_vector, self.probability_vector = None, None
        for goal, pos_class_name in self._data.index.values:
            self._data['confusion'][goal][pos_class_name], self._data['auc'][goal][pos_class_name] = None, 0.  
            self._data['fixed_score'][goal][pos_class_name], self._data['refer_score'][goal][pos_class_name] = float(goal), None
            self._data['tag'][goal][pos_class_name] = ''
    
    def update(self, actual_vector, probability_vector,
               set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):
        # Setting up for use with `pycm` 
        if type(actual_vector[0]) in [list, np.ndarray]: 
            if type(actual_vector[0]) == np.ndarray: actual_vector = actual_vector.tolist()
            actual_vector = [a.index(1.) for a in actual_vector]
        elif type(actual_vector[0]) == 'str': actual_vector = [self.classes.index(a) for a in actual_vector]
               
        # Generating a confusion matrix with predetermined scores.
        roc_dict, roc_fig = module_plottable_metric.ROC_OvO(labels=np.array(actual_vector), probs=np.array(probability_vector), 
                                                        classes=np.unique(actual_vector).tolist(), 
                                                        positive_class_indices=list(self.positive_class_indices.keys()), 
                                                        negative_class_indices=list(self.negative_class_indices.keys()),
                                                        return_result=True)
        
        for goal in self.goal_score:
            if goal > 1 or goal <= 0: print('Warring: Goal score should be less than 1.')
            goal2fpr = 1-goal # spec+fpr = 1
            for pos_class_idx, pos_class_name in self.positive_class_indices.items():
                for neg_class_idx, neg_class_name in self.negative_class_indices.items():
                    try: pos_neg_idx = roc_dict['pos_neg_idx'].index([pos_class_idx, neg_class_idx])
                    except: raise ValueError(f'No ROC was calculated with positive class {pos_class_name} and negative class {neg_class_name}.')
                    fpr, tpr = roc_dict['fpr'][pos_neg_idx], roc_dict['tpr'][pos_neg_idx] # 1 -> 0
                    
                    # If no instances meet the target score, it will return closest_value. 
                    target_fpr, closest_fpr = round(goal2fpr, self.goal_digit), None
                    same_value_index  = np.where(np.around(fpr, self.goal_digit) == target_fpr)[0]
                    if len(same_value_index) == 0:
                        closest_fpr = fpr[np.abs(fpr - target_fpr).argmin()]
                        same_value_index = np.where(fpr == closest_fpr)[0]
                    # Select the value with the highest TPR at the same specificity. (The earlier the index, the higher the score).
                    same_value_index.sort()
                    best_idx = same_value_index[0]
                    print(f'Now goal is {goal} of {pos_class_name} ({same_value_index})')
                    print(f"-> best_idx is {best_idx} & threshold cnt is {len(roc_dict['thresholds'])} (fpr: {len(roc_dict['fpr'])}, tpr: {len(roc_dict['tpr'])})")
                    
                    all_mask = np.logical_or(labels == pos_class_idx, labels == neg_class_idx)
                    all_idx, use_classes = np.flatnonzero(all_mask), [False, True]
                    best_confusion = self._createConfusionMatrixobj(pos_mask[all_mask], probs[all_idx, pos_class_idx], roc_dict['thresholds'][best_idx], pos_class_idx)
                    self._data.confusion[goal][pos_class_name][neg_class_name] = deepcopy(best_confusion)
                    self._data.auc[goal][pos_class_name][neg_class_name] = crv.area()[pos_class_idx]
                    self._data.refer_score[goal][pos_class_name][neg_class_name] = tpr[best_idx]
                    self._data.tag[goal][pos_class_name][neg_class_name] = f'FixedSpec-{str(goal).replace("0.", "")}_Positive-{pos_class_name}_Negative-{neg_class_name}'
                    
                    if img_update or set_title is not None or img_save_dir_path is not None:
                        confusion_plt = self.createConfusionMatrix(goal, pos_class_name)
                        if set_title is None: 
                            set_title = f'Confusion matrix - Fixed Spec: {goal}\n(Positive class: {pos_class_name} VS Negative class: {neg_class_name})'
                        confusion_plt.ax_.set_title(set_title)
                    use_tag = f'ConfusionMatrix:{self.get_tag(goal, pos_class_name, neg_class_name)}'
                    if self.writer is not None and img_update:
                        self.writer.add_figure(use_tag, confusion_plt.figure_)
                    if img_save_dir_path is not None:
                        confusion_plt.figure_.savefig(Path(img_save_dir_path) / use_tag, dpi=300, bbox_inches='tight')

    def _createConfusionMatrixobj(self, actual_vector, probability_vector, threshold, positive_class_idx):
        actual_classes = np.unique(actual_vector).tolist()
        positive_class = actual_classes[positive_class_idx]
        def lambda_fun(x): return threshold_func(x, positive_class, actual_classes, threshold)
        return pycmCM(actual_vector, probability_vector, threshold=lambda_fun)    
    
    def get_confusion_obj(self, goal, pos_class_name, neg_class_name):
        return self._data.confusion[goal][pos_class_name][neg_class_name]
    def get_auc(self, goal, pos_class_name, neg_class_name):
        return self._data.auc[goal][pos_class_name][neg_class_name]
    def get_fixed_score(self, goal, pos_class_name, neg_class_name):
        return self._data.fixed_score[goal][pos_class_name][neg_class_name]
    def get_refer_score(self, goal, pos_class_name, neg_class_name):
        return self._data.refer_score[goal][pos_class_name][neg_class_name]
    def get_tag(self, goal, pos_class_name, neg_class_name):
        return self._data.tag[goal][pos_class_name][neg_class_name]
    def result(self):
        result_data = deepcopy(self._data)
        for goal, pos_class_name, neg_class_name  in result_data.index.values:
            if result_data['confusion'][goal][pos_class_name][neg_class_name] is not None:
                result_data['confusion'][goal][pos_class_name][neg_class_name] = result_data['confusion'][goal][pos_class_name][neg_class_name].to_array().tolist()
            else: result_data['confusion'][goal][pos_class_name][neg_class_name] = None
        return dict(result_data)
    
    def createConfusionMatrix(self, goal, pos_class_name, neg_class_name): 
        cm = self.get_confusion_obj(goal, pos_class_name, neg_class_name).to_array()
        disp = ConfusionMatrixDisplay(cm, display_labels=self.classes)
        confusion_plt = disp.plot(cmap=plt.cm.binary)
        return confusion_plt