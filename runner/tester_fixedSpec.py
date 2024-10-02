import numpy as np
from .tester import Tester
from base import FixedSpecConfusionTracker
from copy import deepcopy
from utils import ensure_dir, write_dict2json, check_and_import_library
from utils import plot_confusion_matrix_1, plot_performance_1, close_all_plots

class FixedSpecTester(Tester):
    """
    Tester class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader, ROCNameForFixedSpec='ROC_OvO'):
        # Removing duplicate AUC calculation since the trainer already computes it.
        if 'fixed_goal' not in config['trainer'].keys():
            raise ValueError('There is no fixed specificity score to track.')
        self.ROCNameForFixedSpec, fixedSpecType = ROCNameForFixedSpec, ROCNameForFixedSpec.split('_')[-1]
        self.AUCNameForFixedSpec, self.AUCNameForReference = f'AUC_{fixedSpecType}', f'AUC_{"OvR" if fixedSpecType=="OvO" else "OvO"}'
        self.AUCForReferenceftns, metrics_module = f'{self.AUCNameForReference}_class', check_and_import_library('model.metric')
        if self.AUCForReferenceftns not in dir(metrics_module):
            raise ValueError(f'Warring: {self.AUCForReferenceftns} is not in the model.metric library.')
        self.AUCForReferenceftns = getattr(metrics_module, self.AUCForReferenceftns)
        _metric_ftns = deepcopy(metric_ftns)
        for met in metric_ftns:
            if any(auc_name.lower() in met.__name__.lower() for auc_name in [self.AUCNameForFixedSpec, self.AUCNameForReference]):
                _metric_ftns.remove(met)
                del config['metrics'][met.__name__]
        metric_ftns = _metric_ftns
        
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader)
        self.config = config
        self.device = device
        
        self.ROCForFixedSpecParams, self.original_result_name  = None, 'maxprob'
        if self.plottable_metrics_kwargs is not None:
            if self.ROCNameForFixedSpec in self.plottable_metrics_kwargs.keys(): 
                self.ROCForFixedSpecParams ={
                    'goal_score':config['trainer']['fixed_goal'],
                    'negative_class_indices':self.plottable_metrics_kwargs[self.ROCNameForFixedSpec]['negative_class_indices'],
                }
                self.test_ROCForFixedSpec = FixedSpecConfusionTracker(goal_score=self.ROCForFixedSpecParams['goal_score'], classes=self.classes,
                                                                       negative_class_indices=self.ROCForFixedSpecParams['negative_class_indices'])
                self.best_auc = {f'{pos_class_name} VS {neg_class_name}':None for goal, pos_class_name, neg_class_name in self.test_ROCForFixedSpec.index}
            else: raise ValueError(f'Warring: {self.ROCNameForFixedSpec} is not in the config[plottable_metrics]')
        else: raise ValueError('Warring: plottable_metrics is not in the config')
                
    def _get_a_log(self):
        '''
        return log
        log = {
            'epoch':1,
            'loss':0.5,
            'val_loss':1.0,
            'auc':{class_name:0.9},
            'val_auc':{class_name:0.7},
            'maxprob':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... },
            'Fixed_spec_goal':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... }
        }
        '''
        basic_log, basic_confusion = self.metrics.result(), self.confusion.result()
        
        # Update Confusion Matrix for FixedSpec
        self.test_ROCForFixedSpec.update(self.confusion.get_actual_vector(self.confusion_key), 
                                         self.confusion.get_probability_vector(self.confusion_key), img_update=False) 
        
        # Basic Result
        log, original_log = {}, {}
        for k, v in basic_log.items():
            if any(basic in k for basic in self.basic_metrics): log[k] = v
            else: original_log[k] = v
        original_log.update(basic_confusion)
        
        # AUC Result
        log.update(self._get_auc())
        
        # Original Result (MaxProb)
        log[self.original_result_name] = original_log
        
        # Goal Result (FixedSpec)
        log.update(self._summarize_ROCForFixedSpec())
            
        self.test_ROCForFixedSpec.reset()
        return log
    
    def _get_auc(self):
        # The AUC is calculated in the `_get_auc` function only for what is set via self.AUCNameForFixedSpec.
        auc_metrics = {}
        # 1. AUC calculated from the ROC curve, which is used for a fixed specificity.
        auc_metrics[self.AUCNameForFixedSpec], use_pair = {}, []
        for goal, pos_class_name, neg_class_name in self.test_ROCForFixedSpec.index:
            if (pos_class_name, neg_class_name) in use_pair: continue
            use_pair.append((pos_class_name, neg_class_name))
            use_tag = f'P-{pos_class_name}_N-{neg_class_name}'
            auc_metrics[self.AUCNameForFixedSpec][use_tag] = self.test_ROCForFixedSpec.get_auc(goal, pos_class_name, neg_class_name)
        
        # 2. Reference AUC to be calculated in other ways 
        auc_metrics[self.AUCNameForReference] = self.AUCForReferenceftns(self.confusion.get_confusion_obj(self.confusion_key), self.classes, method='roc')       
        
        return auc_metrics
    
    def _summarize_ROCForFixedSpec(self):        
        goal_metrics = {}
        # 1. AUCs : Pass (The AUC is calculated in the `_get_auc` function only for what is set via self.AUCNameForFixedSpec.)
        # 2. Other metrics
        confusion_dict = self.test_ROCForFixedSpec.result()
        for goal, pos_class_name, neg_class_name in self.test_ROCForFixedSpec.index:
            pos_class_idx, neg_class_idx = np.where(np.array(self.classes) == pos_class_name)[0][0], np.where(np.array(self.classes) == neg_class_name)[0][0]
            category = self.test_ROCForFixedSpec.get_tag(goal, pos_class_name, neg_class_name)
            confusion_obj = self.test_ROCForFixedSpec.get_confusion_obj(goal, pos_class_name, neg_class_name)
            confusion_classes = np.array(confusion_obj.classes)
            goal_metrics[category] = {}
            for met_idx, met in enumerate(self.metric_ftns): # pycm version
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
                use_confusion_obj = deepcopy(confusion_obj)
                if met_kwargs is None: goal_metrics[category][tag] = met(use_confusion_obj, self.classes)
                else:
                    run = True
                    for key, value in met_kwargs.items():
                        if 'idx' in key:
                            if value not in [pos_class_idx, neg_class_idx]: 
                                run = False
                                break
                        elif 'indices' in key: 
                            valid_indices = [index for index in value if index in [pos_class_idx, neg_class_idx]]
                            if valid_indices == []:
                                run = False
                                break
                            else: met_kwargs[key] = valid_indices
                    if not run: continue
                    goal_metrics[category][tag] = met(use_confusion_obj, self.classes, **met_kwargs)             
            goal_metrics[category][self.confusion_key] = confusion_dict[(goal, pos_class_name, neg_class_name)]
        return goal_metrics
    
    def _save_output(self, log):
        '''
        log = {
            'epoch':1,
            'loss':0.5,
            'val_loss':1.0,
            'auc':{class_name:0.9},
            'val_auc':{class_name:0.7},
            'maxprob':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... },
            'Fixed_spec_goal':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... }
        }
        '''
        basic_log = {key:val for key, val in log.items() if type(val) != dict} # epoch, loss, val_loss, runtime
        auc_log = {key:val for key, val in log.items() if any(auc_name.lower() in key.lower() for auc_name in [self.AUCNameForFixedSpec, self.AUCNameForReference])}
        
        for category, content in log.items():
            if type(content) != dict or any(auc_name.lower() in category.lower() for auc_name in [self.AUCNameForFixedSpec, self.AUCNameForReference]): continue # basic_log, auc_log
            save_metrics_path = self.output_metrics
            if category != self.original_result_name: save_metrics_path = str(save_metrics_path).replace('.json', f'_{category}.json')
            
            # Save the result of metrics.
            result = deepcopy(basic_log)
            result.update(deepcopy(auc_log))
            for k, v in content.items(): result[k] = v
            write_dict2json(result, save_metrics_path)

            # Save the result of confusion matrix image.
            self._make_a_confusion_matrix(content[self.confusion_key], save_mode=f'Test {category}', save_dir=self.confusion_img_dir)
            # Save the reuslt of metrics graphs.
            if self.save_performance_plot: plot_performance_1(result, self.metrics_img_dir/f'metrics_graphs_{category}.png')
            close_all_plots()
            
    def _save_tensorboard(self, log):
        if self.tensorboard:
            for key, value in log.items():
                self.writer.set_step(self.test_epoch, self.wirter_mode)
                if any(item in key.lower() for item in ['epoch', 'confusion', 'time']): continue
                if type(value) != dict: # loss
                    self.writer.add_scalar(key, value)
                elif any(auc_name.lower() in key.lower() for auc_name in [self.AUCNameForFixedSpec, self.AUCNameForReference]): # auc
                    self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
                else: # maxprob, Fixed_spec_goal
                    for new_key, new_value in value.items():
                        if any(item in new_key.lower() for item in ['epoch', 'loss', 'confusion', 'time']): continue
                        if not isinstance(new_value, dict): # 1. All metrics
                            self.writer.add_scalar(new_key, new_value)
                        else: # 2. All metrics per class
                            self.writer.add_scalars(new_key, {str(k):v for k, v in new_value.items()})
            
                    # 3. Confusion Matrix
                    self.writer.set_step(self.test_epoch, f'{self.wirter_mode}_{key}', False)
                    self.writer.add_figure('ConfusionMatrix', self._make_a_confusion_matrix(value[self.confusion_key]))
                    close_all_plots()