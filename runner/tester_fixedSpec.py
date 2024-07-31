from .tester import Tester
from base import FixedSpecConfusionTracker
from copy import deepcopy
from utils import ensure_dir, write_dict2json, plot_confusion_matrix_1, plot_performance_1, close_all_plots

class FixedSpecTester(Tester):
    """
    Tester class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader):
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader)
        self.config = config
        self.device = device
        
        # Removing duplicate AUC calculation since the trainer already computes it.
        if 'fixed_goal' not in config['trainer'].keys():
            raise ValueError('There is no fixed specificity score to track.')
        self.ROCNameForFixedSpec, self.AUCNameForFixedSpec = 'ROC_OvO', 'AUC_OvO'
        for met in self.metric_ftns:
            if self.AUCNameForFixedSpec.lower() in met.__name__.lower(): self.metric_ftns.remove(met)
            
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
            'self.AUCNameForFixedSpec':{class_name:0.9},
            'val_self.AUCNameForFixedSpec':{class_name:0.7},
            'maxprob':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... },
            'Fixed_spec_goal':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... }
        }
        '''
        log = self.metrics.result()
        log_confusion = self.confusion.result()
        
        # Original Result (MaxProb)
        log[self.original_result_name]={}
        for met in self.metric_ftns:
            log[self.original_result_name][met.__name__] = log[met.__name__] 
            del log[met.__name__]
        log[self.original_result_name].update(log_confusion)
        
        # Goal Result (FixedSpec)
        log.update(self._summarize_ROCForFixedSpec())
        
        # AUC Result
        log[self.AUCNameForFixedSpec], use_pair = {}, []
        for goal, pos_class_name, neg_class_name in self.test_ROCForFixedSpec.index:
            if (pos_class_name, neg_class_name) in use_pair: continue
            use_pair.append((pos_class_name, neg_class_name))
            use_tag = f'P-{pos_class_name}_N-{neg_class_name}'
            log[self.AUCNameForFixedSpec][use_tag] = self.test_ROCForFixedSpec.get_auc(goal, pos_class_name, neg_class_name)
            
        self.test_ROCForFixedSpec.reset()
        return log
    
    def _summarize_ROCForFixedSpec(self):        
        self.test_ROCForFixedSpec.update(self.confusion.get_actual_vector(self.confusion_key), 
                                         self.confusion.get_probability_vector(self.confusion_key), img_update=False) 
        goal_metrics = {}
        # 1. AUC: Pass
        # 2. Metrics
        confusion_dict = self.test_ROCForFixedSpec.result()
        for goal, pos_class_name, neg_class_name in self.test_ROCForFixedSpec.index:
            category = self.test_ROCForFixedSpec.get_tag(goal, pos_class_name, neg_class_name)
            confusion_obj = self.test_ROCForFixedSpec.get_confusion_obj(goal, pos_class_name, neg_class_name)
            goal_metrics[category] = {}
            for met in self.metric_ftns:# pycm version
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]))
                tag = met.__name__ if tag is None else tag
                use_confusion_obj = deepcopy(confusion_obj)                             
                if met_kwargs is None: goal_metrics[category][tag] = met(use_confusion_obj, self.classes)
                else: goal_metrics[category][tag] = met(use_confusion_obj, self.classes, **met_kwargs)                
            goal_metrics[category][self.confusion_key] = confusion_dict[(goal, pos_class_name, neg_class_name)]
        return goal_metrics
    
    def _save_output(self, log):
        '''
        log = {
            'epoch':1,
            'loss':0.5,
            'val_loss':1.0,
            'self.AUCNameForFixedSpec':{class_name:0.9},
            'val_self.AUCNameForFixedSpec':{class_name:0.7},
            'maxprob':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... },
            'Fixed_spec_goal':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... }
        }
        '''
        basic_log = {key:val for key, val in log.items() if type(val) != dict} # epoch, loss, val_loss, runtime
        auc_log = {key:val for key, val in log.items() if self.AUCNameForFixedSpec in key}
        
        for category, content in log.items():
            if type(content) != dict or self.AUCNameForFixedSpec in category: continue # basic_log, auc_log
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
                if key in ['epoch', 'confusion'] or 'time' in key: continue
                if type(value) != dict: # loss
                    self.writer.add_scalar(key, value)
                elif self.AUCNameForFixedSpec in key:
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
                    self.writer.add_figure('ConfusionMatrix', self._make_a_confusion_matrix(value[self.confusion_key]))
                    close_all_plots()