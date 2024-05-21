from .tester import Tester
from base import FixedSpecConfusionTracker
from copy import deepcopy
from utils import write_dict2json, plot_confusion_matrix_1, plot_performance_1, plot_close

class FixedSpecTester(Tester):
    """
    Tester class
    """
    def __init__(self, model, criterion, metric_ftns, curve_metric_ftns, config, classes, device, data_loader):
        super().__init__(model, criterion, metric_ftns, curve_metric_ftns, config, classes, device, data_loader)
        self.config = config
        self.device = device
        
        # Removing duplicate AUC calculation since the trainer already computes it.
        for met in self.metric_ftns:
            if met.__name__.lower() == 'auc': self.metric_ftns.remove(met)
            
        curve_metrics = self.config.config['curve_metrics'] if 'curve_metrics' in self.config.config.keys() else  None
        self.FixedNegativeROC, self.original_result_name = None, 'maxprob'
        self.auc = {class_name:None for class_name in self.classes}
        if curve_metrics is not None:
            if 'FixedNegativeROC' in curve_metrics.keys(): 
                self.FixedNegativeROC ={
                    'goal_score':curve_metrics['FixedNegativeROC']['fixed_goal'],
                    'negative_class_idx':curve_metrics['FixedNegativeROC']['negative_class_idx'],
                    'output_dir':self.output_dir / f"{curve_metrics['FixedNegativeROC']['save_dir']}"
                }
                self.goal_for_auc = self.FixedNegativeROC['goal_score'][0]
                self.FixedNegativeROC['output_metrics'] = self.FixedNegativeROC['output_dir'] / 'metrics-test.json'
                if not self.FixedNegativeROC['output_dir'].is_dir(): self.FixedNegativeROC['output_dir'].mkdir(parents=True, exist_ok=True)
                self.test_FixedNegativeROC = FixedSpecConfusionTracker(goal_score=self.FixedNegativeROC['goal_score'], classes=self.classes,
                                                                       negative_class_idx=self.FixedNegativeROC['negative_class_idx'])
            else: raise ValueError('Warring: FixedNegativeROC is not in the config[curve_metrics]') 
        else: print('Warring: curve_metrics is not in the config')
    
    def _curve_metrics(self):
        for met in self.curve_metric_ftns:
            if met.__name__ == 'FixedNegativeROC':
                curve_fig = met(self.confusion.get_actual_vector(self.confusion_key),
                                self.confusion.get_probability_vector(self.confusion_key), 
                                self.classes, self.FixedNegativeROC['negative_class_idx'])
            else: curve_fig = met(self.confusion.get_actual_vector(self.confusion_key),
                                  self.confusion.get_probability_vector(self.confusion_key), self.classes)
            self.writer.add_figure(met.__name__, curve_fig)
            if self.save_performance_plot: curve_fig.savefig(self.output_dir / f'{met.__name__}.png', bbox_inches='tight')
                
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
        log = self.metrics.result()
        log[self.original_result_name]={}
        for met in self.metric_ftns:
            log[self.original_result_name][met.__name__] = log[met.__name__] 
            del log[met.__name__]
        
        log_confusion = self.confusion.result()
        log[self.original_result_name].update(log_confusion)
        
        log.update(self._FixedNegativeROCResult())
        log['auc'] = {}
        for pos_class_idx, pos_class_name in self.test_FixedNegativeROC.positive_classes.items():
            log['auc'][pos_class_name] = self.test_FixedNegativeROC.get_auc(self.goal_for_auc, pos_class_name)
            
        self.test_FixedNegativeROC.reset()
        return log
    
    def _FixedNegativeROCResult(self):        
        img_save_dir_path = str(self.FixedNegativeROC['output_dir']) if self.save_performance_plot else None
        self.test_FixedNegativeROC.update(self.confusion.get_actual_vector(self.confusion_key), 
                                          self.confusion.get_probability_vector(self.confusion_key), 
                                          img_update=True, img_save_dir_path=img_save_dir_path)
        goal_metrics = {}
        # 1. AUC: Pass
        # 2. Metrics
        for goal, pos_class_name in self.test_FixedNegativeROC.index:
            use_key = f'fixedSpec_{goal*100:.0f}_Positive_{pos_class_name}'
            goal_metrics[use_key] = {}
            confusion_obj = self.test_FixedNegativeROC.get_confusion_obj(goal, pos_class_name)
            
            for met in self.metric_ftns:# pycm version
                met_name_idx = self.metrics_class_index[met.__name__]
                if met_name_idx is None: goal_metrics[use_key][met.__name__] = met(confusion_obj, self.classes)
                else: goal_metrics[use_key][met.__name__] = met(confusion_obj, self.classes, met_name_idx)
            goal_metrics[use_key][self.confusion_key] = confusion_obj.to_array().tolist()
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
        auc_log = {key:val for key, val in log.items() if 'auc' in key}
        
        for key, val in log.items():
            if type(val) != dict or 'auc' in key: continue # basic_log, auc_log
            
            # Save the result of confusion matrix image.
            if key == self.original_result_name:
                plot_confusion_matrix_1(val[self.confusion_key], self.classes, 
                                        'Confusion Matrix: Test Data', self.output_dir/f'confusion_matrix_test.png')
            
            # Save the result of metrics.
            save_metrics_path = self.output_metrics if key == self.original_result_name else str(self.FixedNegativeROC['output_metrics']).replace('.json', f'_{key}.json')
            result = deepcopy(basic_log)
            result.update(deepcopy(auc_log))
            for k, v in val.items(): result[k] = v
            write_dict2json(result, save_metrics_path)

            # Save the reuslt of metrics graphs.
            save_dir = self.output_dir if key == self.original_result_name else self.FixedNegativeROC['output_dir']
            if self.save_performance_plot: plot_performance_1(result, save_dir/f'metrics_graphs_{key}.png')
            
            plot_close()
            
    def _save_tensorboard(self, log):
        if self.tensorboard:
            for key, value in log.items():
                self.writer.set_step(self.test_epoch, self.wirter_mode)
                if key in ['epoch', 'confusion'] or 'time' in key: continue
                if type(value) != dict: # loss
                    self.writer.add_scalar(key, value)
                elif 'auc' in key:
                    self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
                else: # maxprob, Fixed_spec_goal
                    for new_key, new_value in value.items():
                        if new_key in ['epoch', 'confusion'] or 'time' in new_key: continue
                        # 1. All metrics
                        if '_class' not in new_key: self.writer.add_scalar(new_key, new_value)
                        # 2. All metrics per class
                        else: self.writer.add_scalars(new_key, {str(k):v for k, v in new_value.items()})
            
                    # 3. Confusion Matrix
                    self.writer.set_step(self.test_epoch, f'{self.wirter_mode}_{key}')
                    self.writer.add_figure('ConfusionMatrix', plot_confusion_matrix_1(value['confusion'], self.classes, return_plot=True))
                    plot_close()