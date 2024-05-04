import os
import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay
from pycm import ConfusionMatrix as pycmCM
from pycm import ROCCurve
import model.metric as module_metric

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
    def __init__(self, classes, goal_score:list, negative_class_idx:int, writer=None):
        self.writer = writer
        self.classes = classes
        
        self.fixed_metrics_ftns = getattr(module_metric, 'specificity')
        self.refer_metrics_ftns = getattr(module_metric, 'sensitivity')
        self.negative_class_idx = negative_class_idx
        self.positive_classes = {class_idx:class_name for class_idx, class_name in enumerate(self.classes) if class_idx != self.negative_class_idx}
        self.goal_score = goal_score 
        
        self._data = pd.DataFrame(index=goal_score, columns=['confusion', 'threshold', 'fixed_score', 'refer_score', 'refer_loss', 'best'])
        self.reset()
    
    def reset(self):
        self.actual_vector, self.probability_vector = None, None
        for key in self._data.index.values:
            self._data['confusion'][key], self._data['best'][key] = None, False  
            self._data['threshold'][key], self._data['fixed_score'][key], = 1., float(key)
            self._data['refer_score'][key], self._data['refer_loss'][key] = None, np.inf
        
    def update(self, actual_vector, probability_vector, loss,
               set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):
        # Setting up for use with `pycm` 
        if type(actual_vector[0]) in [list, np.ndarray]: 
            if type(actual_vector[0]) == np.ndarray: actual_vector = actual_vector.tolist()
            actual_vector = [a.index(1.) for a in actual_vector]
        elif type(actual_vector[0]) == 'str': actual_vector = [self.classes.index(a) for a in actual_vector]
               
        # Generating a confusion matrix with predetermined scores.
        crv = ROCCurve(actual_vector=np.array(actual_vector), probs=np.array(probability_vector), classes=np.unique(actual_vector).tolist())
        
        # ROCCurve에서는 thresholds의 인덱스 기준으로 FPR과 TPR을 반환함.
        # 그러나 FPR은 맨 뒤에 0, TPR은 맨 앞에 1이 하나 더 삽입된 상태로 반환됨.
        # 그리고 FPR의 경우 뒤집힌 상태로 반환되어, 가장 적절한 값의 index를 사용하기 위해 뒤집음
        fpr = np.flip(np.delete(np.array(crv.data[self.negative_class_idx]['FPR']), -1))
        tpr = {class_idx:np.delete(np.array(crv.data[class_idx]['TPR']), 0) for class_idx in self.positive_classes.keys()}

        for goal in self.goal_score:
            if goal > 1: print('Warring: Goal score should be less than 1.')
            # If no instances meet the target score, it will return closest_value. 
            target_fpr, closest_fpr = round(1-goal, 2), None
            same_value_index  = np.where(np.around(fpr, 2) == target_fpr)[0]
            if len(same_value_index) == 0:
                # print('Find the closest value')
                closest_fpr = fpr[np.abs(fpr - target_fpr).argmin()]
                same_value_index = np.where(fpr == closest_fpr)[0]
            
            best_idx = None
            for goal_index in same_value_index:
                if best_idx is None: best_idx = goal_index                
                elif fpr[best_idx] == fpr[goal_index]:
                    now_item_is_best = [pos_class_idx for pos_class_idx in tpr.keys() if tpr[pos_class_idx][best_idx] < tpr[pos_class_idx][goal_index]]
                    if len(now_item_is_best) > len(self.classes)/2: best_idx = goal_index 
                elif fpr[best_idx] < fpr[goal_index]: best_idx = goal_index 
            
             # Evaluating for optimal performance by analyzing various metrics from the confusion matrix with predefined scores.
            if self._data.refer_score[goal] is not None: # target_fpr
                refer_score = self._data.refer_score[goal]
                if type(refer_score) == dict: refer_score = np.mean(list(refer_score.values()))
                if refer_score == fpr[best_idx]:
                    if loss < self._data.refer_loss[goal]: self._data.best[goal] = True
                elif refer_score < fpr[best_idx]: self._data.best[goal] = True
            elif closest_fpr is None: self._data.best[goal] = True
            
            best_confusion = self._createConfusionMatrixobj(actual_vector, probability_vector, crv.thresholds[best_idx])
            self._data.confusion[goal] = deepcopy(best_confusion)
            self._data.threshold[goal] = deepcopy(crv.thresholds[best_idx])
            self._data.refer_score[goal] = {class_idx:tpr[class_idx][best_idx] for class_idx in self.positive_classes.keys()}
            self._data.refer_loss[goal] = deepcopy(loss)
            
            if img_update or set_title is not None or img_save_dir_path is not None:
                confusion_plt = self.createConfusionMatrix(goal)
                confusion_plt.ax_.set_title(set_title if set_title is not None else f'Confusion matrix - Fixed Spec: {goal}')
            use_tag = f'ConfusionMatrix_FixedSpec_{str(goal).replace("0.", "")}'
            if self.writer is not None and img_update:
                self.writer.add_figure(use_tag, confusion_plt.figure_)
            if img_save_dir_path is not None:
                confusion_plt.figure_.savefig(Path(img_save_dir_path) / use_tag, dpi=300, bbox_inches='tight')

    def _createConfusionMatrixobj(self, actual_vector, probability_vector, threshold):
        actual_prob_vector = [p[a] for p, a in zip(probability_vector, actual_vector)]       
        use_pred, use_prob = np.array(actual_prob_vector > threshold), np.array(deepcopy(probability_vector))
        use_prob[np.arange(len(actual_vector)), actual_vector] = -np.inf
        use_pred = np.where(use_pred, actual_vector, np.argmax(use_prob, axis=1))
        return pycmCM(actual_vector, use_pred)    
    
    def get_confusion_obj(self, key):
        return self._data.confusion[key]
    def get_threshold(self, key):
        return self._data.threshold[key]
    def get_fixed_score(self, key):
        return self._data.fixed_score[key]
    def get_refer_score(self, key):
        return self._data.refer_score[key]
    def get_refer_loss(self, key):
        return self._data.refer_loss[key]
    def result(self):
        result_data = deepcopy(self._data)
        for key in result_data.index.values:
            result_data['confusion'][key] = result_data['confusion'][key].to_array().tolist() if result_data['confusion'][key] is not None else None
        return dict(result_data)
    
    def createConfusionMatrix(self, key): 
        disp = ConfusionMatrixDisplay(self._data.confusion[key].to_array(), display_labels=self.classes)
        confusion_plt = disp.plot(cmap=plt.cm.binary)
        return confusion_plt