import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path
from pycm import ConfusionMatrix as pycmCM
from pycm.pycm_util import threshold_func
import model.plottable_metrics  as module_plottable_metric
from utils import integer_encoding

class ConfusionTracker:
    def __init__(self, *keys, classes, writer=None):
        self.writer = writer
        self.classes = classes        
        self._data = pd.DataFrame(index=keys, columns=['actual', 'predict', 'probability', 'confusion'])
        self.index = self._data.index.values
        self.reset()

    def reset(self):
        self._data.loc[:, 'confusion'] = None
        for key in self.index:
            self._data.loc[key, 'actual'] = []
            self._data.loc[key, 'predict'] = []
            self._data.loc[key, 'probability'] = []

    def update(self, key, value:dict, set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):
        required_keys = ['actual', 'predict', 'probability']
        if not all(k in list(value.keys()) for k in required_keys):
            # if 'actual' not in list(value.keys()) or 'predict' not in list(value.keys()):
            raise ValueError('Correct answer (actual), predicted (predict) and probability value (probability) are required to update ConfusionTracker.'
                             + f'\nNow Value {list(value.keys())}.')
        if np.array(value['probability']).ndim != 2:
            raise ValueError(f'Probability value (probability) should be a 2D array. Now shape is {np.array(value["probability"]).shape}.')
        self._data.loc[key, 'actual'].extend([self.classes[class_idx] for class_idx in integer_encoding(value['actual'], self.classes)])
        self._data.loc[key, 'predict'].extend([self.classes[class_idx] for class_idx in integer_encoding(value['predict'], self.classes)])
        self._data.loc[key, 'probability'].extend(value['probability'])
        
        # A basic confusion matrix is generated based on the class with the highest probability.
        cm = pycmCM(actual_vector=np.array(self._data.loc[key, 'actual']), predict_vector=np.array(self._data.loc[key, 'predict']))
        cm.prob_vector = self._data.loc[key, 'probability']
        self._data.loc[key, 'confusion'] = deepcopy(cm)

        if img_update or set_title is not None or img_save_dir_path is not None:
            # Perform only when all classes of data are present         
            self.plotConfusionMatrix(key, set_title, img_save_dir_path, img_update)
        
    def get_actual_vector(self, key):
        return self._data.loc[key, 'actual']
    def get_prediction_vector(self, key):
        return self._data.loc[key, 'predict']
    def get_probability_vector(self, key):
        return self._data.loc[key, 'probability']
    def get_confusion_obj(self, key):
        return self._data.loc[key, 'confusion']
    def get_confusion_matrix(self, key, return_type:type=list):
        cm = self.get_confusion_obj(key)
        if type(cm) != pycmCM: return None
        if return_type==np.ndarray: return cm.to_array()
        elif return_type==list: return cm.to_array().tolist()
        elif return_type==dict: return cm.table
        else: TypeError(f'Unsupported return type: {return_type}')
    def result(self):
        return {key:self.get_confusion_matrix(key, dict) for key in self.index}

    def plotConfusionMatrix(self, key, 
                            title:str=None, img_save_dir_path=None, img_update:bool=False, return_plot:bool=False):
        cm = self.get_confusion_obj(key)
        if type(cm) != pycmCM: raise TypeError(f'It is not a pycm object.')
        if title is None: title = f'Confusion matrix: {key}'
        confusion_plt = cm.plot(number_label=True, title=title).figure
        use_tag = f'ConfusionMatrix_{key}'
        if self.writer is not None and img_update:
            self.writer.add_figure(use_tag, confusion_plt)
        if img_save_dir_path is not None:
            confusion_plt.savefig(Path(img_save_dir_path)/use_tag, dpi=300, bbox_inches='tight')
        if return_plot: return confusion_plt
    
    def saveConfusionMatrix(self, key, save_dir, save_name:str='cm'): 
        cm = self.get_confusion_obj(key)
        if type(cm) != pycmCM: print('Warning: Can\'t save because there is no confusion matrix.')
        if 'cm' not in save_name: save_name = f'cm_{save_name}'
        cm.save_obj(str(Path(save_dir)/save_name), save_stat=False, save_vector=True)
        
    
class FixedSpecConfusionTracker:
    """ The current metric uses the one-vs-one strategy. """
    def __init__(self, classes, goal_score:list, negative_class_indices:[int, list, np.ndarray], goal_digit:int=2, writer=None):
        self.writer = writer
        self.classes = np.array(classes)
        if isinstance(negative_class_indices, int): negative_class_indices = [negative_class_indices]
        self.negative_class_indices = {class_idx:class_name for class_idx, class_name in enumerate(self.classes) if class_idx in negative_class_indices}
        self.positive_class_indices = {class_idx:class_name for class_idx, class_name in enumerate(self.classes) if class_idx not in negative_class_indices}
        if self.positive_class_indices == {}: self.positive_class_indices = deepcopy(self.negative_class_indices) # all classes are negative and positive
        
        self.goal_digit, self.goal_score = goal_digit, goal_score
        self.index = [[], [], []]
        for goal in self.goal_score:
            for pos_class_name in self.positive_class_indices.values():
                for neg_class_name in self.negative_class_indices.values():
                    if pos_class_name == neg_class_name: continue
                    self.index[0].append(goal)
                    self.index[1].append(pos_class_name)
                    self.index[2].append(neg_class_name)
        self._data = pd.DataFrame(index=self.index, columns=['confusion', 'auc', 'fixed_score', 'refer_score', 'tag'])
        self.index = self._data.index.values
        self.reset()
    
    def reset(self):
        self._data.loc[:, 'confusion'] = None
        self._data.loc[:, 'auc'] = 0. 
        self._data.loc[:, 'refer_score'] = None
        self._data.loc[:, 'tag'] = ''
        for goal, p, n in self.index: 
            if goal > 1 or goal <= 0: raise ValueError('Warring: Goal score should be less than 1.')
            self._data.loc[(goal, p, n)] = float(goal)
    
    def update(self, actual_vector, probability_vector,
               set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):   
        if np.array(probability_vector).ndim != 2:
            raise ValueError(f'Probability value (probability) should be a 2D array. Now shape is {np.array(probability_vector).shape}.')  
        actual_vector, probability_vector = integer_encoding(actual_vector, self.classes), np.array(probability_vector)
        actual_classes = np.unique(actual_vector).tolist()
        
        # Generating a confusion matrix with predetermined scores.
        roc_dict, roc_fig = module_plottable_metric.ROC_OvO(labels=actual_vector, probs=probability_vector, classes=self.classes, 
                                                            positive_class_indices=list(self.positive_class_indices.keys()), 
                                                            negative_class_indices=list(self.negative_class_indices.keys()),
                                                            return_result=True)
        for goal, pos_class_name, neg_class_name in self.index:
            goal2fpr = 1-goal # spec+fpr = 1
            try: 
                pos_class_idx, neg_class_idx = np.where(self.classes == pos_class_name)[0][0], np.where(self.classes == neg_class_name)[0][0]
                pos_neg_idx = roc_dict['pos_neg_idx'].index([pos_class_idx, neg_class_idx])
            except: raise ValueError(f'No ROC was calculated with positive class {pos_class_name} and negative class {neg_class_name}.')
            fpr, tpr = roc_dict['fpr'][pos_neg_idx], roc_dict['tpr'][pos_neg_idx] # 1 -> 0
            thresholds = roc_dict['thresholds'][pos_neg_idx] # 0 -> 1
            
            # If no instances meet the target score, it will return closest_value. 
            target_fpr, closest_fpr = round(goal2fpr, self.goal_digit), None
            same_value_index  = np.where(np.around(fpr, self.goal_digit) == target_fpr)[0]
            if len(same_value_index) == 0:
                closest_fpr = fpr[np.abs(fpr - target_fpr).argmin()]
                same_value_index = np.where(fpr == closest_fpr)[0]
            # Select the value with the highest TPR at the same specificity. (The earlier the index, the higher the score).
            same_value_index.sort()
            best_idx = same_value_index[0]
            
            pos_mask, neg_mask = actual_vector == pos_class_idx, actual_vector == neg_class_idx
            all_mask = np.logical_or(pos_mask, neg_mask)
            all_idx, use_classes = np.flatnonzero(all_mask), [False, True]
            pos_labels, pos_probs = pos_mask[all_mask], probability_vector[all_idx, pos_class_idx]
            
            # A basic confusion matrix is generated based on the class with the highest probability.
            pos_labels = [pos_class_name if p else neg_class_name for p in pos_labels]
            best_cm = self._createConfusionMatrixobj(pos_labels, pos_probs, thresholds[best_idx], [neg_class_name, pos_class_name])
            best_cm.prob_vector = {'pos_probs':pos_probs, 'all_probs':probability_vector[all_idx, :], 'pos_class_idx': pos_class_idx, 'neg_class_idx': neg_class_idx}
            self._data.loc[(goal, pos_class_name, neg_class_name), 'confusion'] = deepcopy(best_cm)
            self._data.loc[(goal, pos_class_name, neg_class_name), 'auc'] = roc_dict['auc'][list(self.positive_class_indices.keys()).index(pos_class_idx)]
            self._data.loc[(goal, pos_class_name, neg_class_name), 'refer_score'] = tpr[best_idx]
            self._data.loc[(goal, pos_class_name, neg_class_name), 'tag'] = f'FixedSpec-{str(goal).replace("0.", "")}_Positive-{pos_class_name}_Negative-{neg_class_name}'
            
            if img_update or set_title is not None or img_save_dir_path is not None:
                self.plotConfusionMatrix(goal, pos_class_name, neg_class_name, 
                                         set_title, img_save_dir_path, img_update)

    def _createConfusionMatrixobj(self, actual_vector, probability_vector, threshold, actual_classes):
        def lambda_fun(x): return threshold_func(x, True, actual_classes, threshold)
        return pycmCM(np.array(actual_vector), np.array(probability_vector), threshold=lambda_fun)  
    
    def get_auc(self, goal, pos_class_name, neg_class_name):
        return self._data.loc[(goal, pos_class_name, neg_class_name), 'auc']
    def get_fixed_score(self, goal, pos_class_name, neg_class_name):
        return self._data.loc[(goal, pos_class_name, neg_class_name), 'fixed_score']
    def get_refer_score(self, goal, pos_class_name, neg_class_name):
        return self._data.loc[(goal, pos_class_name, neg_class_name), 'refer_score']
    def get_tag(self, goal, pos_class_name, neg_class_name):
        return self._data.loc[(goal, pos_class_name, neg_class_name), 'tag']
    def get_confusion_obj(self, goal, pos_class_name, neg_class_name):
        return self._data.loc[(goal, pos_class_name, neg_class_name), 'confusion']
    def get_confusion_matrix(self, goal, pos_class_name, neg_class_name, return_type:type=list):
        cm = self.get_confusion_obj(goal, pos_class_name, neg_class_name)
        if type(cm) != pycmCM: return None
        if return_type==np.ndarray: return cm.to_array()
        elif return_type==list: return cm.to_array().tolist()
        elif return_type==dict: return cm.table
        else: TypeError(f'Unsupported return type: {return_type}')
    def result(self):
        return {key:self.get_confusion_matrix(*key, dict) for key in self.index}
    
    def plotConfusionMatrix(self, goal, pos_class_name, neg_class_name, 
                            title:str=None, img_save_dir_path=None, img_update:bool=False): 
        cm = self.get_confusion_obj(goal, pos_class_name, neg_class_name)
        if type(cm) != pycmCM: raise TypeError(f'It is not a pycm object.')
        if title is None: 
            title = f'Confusion matrix - Fixed Spec: {goal}\n'
            title += f'(Positive class: {pos_class_name} VS Negative class: {neg_class_name})'
        confusion_plt = cm.plot(number_label=True, title=title).figure
        use_tag = f'ConfusionMatrix_{self.get_tag(goal, pos_class_name, neg_class_name)}'
        if self.writer is not None and img_update:
            self.writer.add_figure(use_tag, confusion_plt)
        if img_save_dir_path is not None:
            confusion_plt.savefig(Path(img_save_dir_path)/use_tag, dpi=300, bbox_inches='tight')