from .trainer import Trainer
from base import FixedSpecConfusionTracker
from copy import deepcopy
from pathlib import Path
from utils import ensure_dir, read_json, write_dict2json
from utils import plot_confusion_matrix_1, plot_performance_N, close_all_plots
import numpy as np

class FixedSpecTrainer(Trainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, sampling_ftns=None, da_ftns=None):
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device,
                         data_loader, valid_data_loader, lr_scheduler, len_epoch, sampling_ftns, da_ftns)
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
                self.train_ROCForFixedSpec = FixedSpecConfusionTracker(goal_score=self.ROCForFixedSpecParams['goal_score'], classes=self.classes,
                                                                       negative_class_indices=self.ROCForFixedSpecParams['negative_class_indices'])
                self.valid_ROCForFixedSpec = deepcopy(self.train_ROCForFixedSpec)
                self.best_auc = {f'{pos_class_name} VS {neg_class_name}':None for goal, pos_class_name, neg_class_name in self.train_ROCForFixedSpec.index}
            else: raise ValueError(f'Warring: {self.ROCNameForFixedSpec} is not in the config[plottable_metrics]')
        else: raise ValueError('Warring: plottable_metrics is not in the config')
        
    def _get_a_log(self, epoch):
        '''
        return log
        log = {
            'epoch':1,
            'loss':0.5,
            'val_loss':1.0,
            'self.ROCNameForFixedSpec':{class_name:0.9},
            'val_self.ROCNameForFixedSpec':{class_name:0.7},
            'maxprob':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... },
            'Fixed_spec_goal':{'metrics':..., 'val_metrics':..., 'confusion':..., 'val_confusion':... }
        }
        '''
        log, log_confusion = self.train_metrics.result(), self.train_confusion.result()
        if self.do_validation:
            self._valid_epoch(epoch)
            val_log, val_confusion = self.valid_metrics.result(),  self.valid_confusion.result()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log_confusion.update(**{'val_'+k : v for k, v in val_confusion.items()})
        
        # Original Result (MaxProb)
        log[self.original_result_name]={}
        for met in self.metric_ftns:
            log[self.original_result_name][met.__name__] = log[met.__name__] 
            del log[met.__name__]
        if self.do_validation: 
            for met in self.metric_ftns:
                metric_name = f'val_{met.__name__}'
                log[self.original_result_name][metric_name] = log[metric_name] 
                del log[metric_name] 
        log[self.original_result_name].update(log_confusion)
        log[self.original_result_name] = self._sort_train_val_sequences(log[self.original_result_name])
        
        # Goal Result (FixedSpec)
        log.update(self._summarize_ROCForFixedSpec(mode='training'))
        if self.do_validation:
            val_log = self._summarize_ROCForFixedSpec(mode='validation')
            for key, value in val_log.items(): 
                log[key].update(**{'val_'+k : v for k, v in value.items()})
                log[key] = self._sort_train_val_sequences(log[key])
        
        # AUC Result
        log[self.AUCNameForFixedSpec], use_pair = {}, []
        if self.do_validation: log[f'val_{self.AUCNameForFixedSpec}'] = {}
        for goal, pos_class_name, neg_class_name in self.train_ROCForFixedSpec.index:
            if (pos_class_name, neg_class_name) in use_pair: continue
            use_pair.append((pos_class_name, neg_class_name))
            use_tag = f'P-{pos_class_name}_N-{neg_class_name}'
            log[self.AUCNameForFixedSpec][use_tag] = self.train_ROCForFixedSpec.get_auc(goal, pos_class_name, neg_class_name)
            if self.do_validation: log[f'val_{self.AUCNameForFixedSpec}'][use_tag] = self.valid_ROCForFixedSpec.get_auc(goal, pos_class_name, neg_class_name)
        
        # model save and reset
        self._save_BestFixedSpecModel(epoch)
        self.train_ROCForFixedSpec.reset()
        self.valid_ROCForFixedSpec.reset()
        return log
    
    def _summarize_ROCForFixedSpec(self, mode='training'):    
        if mode=='training':
            ROCForFixedSpec = self.train_ROCForFixedSpec
            self.train_ROCForFixedSpec.update(self.train_confusion.get_actual_vector(self.confusion_key),
                                              self.train_confusion.get_probability_vector(self.confusion_key), img_update=False)
            maxprob_confusion = self.train_confusion.get_confusion_obj(self.confusion_key)
        else:
            ROCForFixedSpec = self.valid_ROCForFixedSpec
            self.valid_ROCForFixedSpec.update(self.valid_confusion.get_actual_vector(self.confusion_key),
                                              self.valid_confusion.get_probability_vector(self.confusion_key), img_update=False)
            maxprob_confusion = self.valid_confusion.get_confusion_obj(self.confusion_key)
        goal_metrics = {}
        # 1. AUC: Pass
        # 2. Metrics
        confusion_dict = ROCForFixedSpec.result()
        for goal, pos_class_name, neg_class_name in ROCForFixedSpec.index:
            category = ROCForFixedSpec.get_tag(goal, pos_class_name, neg_class_name)
            confusion_obj = ROCForFixedSpec.get_confusion_obj(goal, pos_class_name, neg_class_name)
            goal_metrics[category] = {}
            for met in self.metric_ftns:# pycm version
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]))
                tag = met.__name__ if tag is None else tag
                use_confusion_obj = deepcopy(confusion_obj) if 'auc' not in met.__name__.lower() else deepcopy(maxprob_confusion)             
                if met_kwargs is None: goal_metrics[category][tag] = met(use_confusion_obj, self.classes)
                else: goal_metrics[category][tag] = met(use_confusion_obj, self.classes, **met_kwargs)                
            goal_metrics[category][self.confusion_key] = confusion_dict[(goal, pos_class_name, neg_class_name)]
        return goal_metrics
    
    def _save_BestFixedSpecModel(self, epoch): 
        # Determines model saving based on improvement in AUC (Area Under Curve) score.
        self.logger.info('')
        ROCForFixedSpec = self.train_ROCForFixedSpec if self.do_validation else self.valid_ROCForFixedSpec
        use_pair = []
        for goal, pos_class_name, neg_class_name in ROCForFixedSpec.index:
            if (pos_class_name, neg_class_name) in use_pair: continue
            use_pair.append((pos_class_name, neg_class_name))
            auc = ROCForFixedSpec.get_auc(goal, pos_class_name, neg_class_name)
            if self.best_auc[f'{pos_class_name} VS {neg_class_name}'] is None or self.best_auc[f'{pos_class_name} VS {neg_class_name}'] < auc:
                self.best_auc[f'{pos_class_name} VS {neg_class_name}'] = auc
                self._save_checkpoint(epoch, filename=f'model_best_AUC_{pos_class_name}VS{neg_class_name}', 
                                      message=f'Saving current best AUC model... ({self.ROCNameForFixedSpec}, {pos_class_name}(+) VS {neg_class_name}(-))')
    
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
            if category != self.original_result_name: save_metrics_path = Path(str(save_metrics_path).replace('.json', f'_{category}.json'))
            
            # Save the result of metrics.
            if save_metrics_path.is_file():
                result = read_json(save_metrics_path) 
                # Adjusting the number of results per epoch.
                result = self._slice_dict_values(result, int(basic_log['epoch'])) 
                # Adding additional information by default.
                result = self._merge_and_append_json(result, basic_log)
                result = self._merge_and_append_json(result, auc_log)
                # Adding additional information.
                result = self._merge_and_append_json(result, content)
            else:
                # Adding additional information by default.
                result = self._convert_values_to_list(basic_log)
                result.update(self._convert_values_to_list(auc_log))
                # Adding additional information.
                result.update(self._convert_values_to_list(content))
            write_dict2json(result, save_metrics_path)
            
            # Save the result of confusion matrix image. 
            self._make_a_confusion_matrix(content[self.confusion_key], save_mode=f'Training {category}', save_dir=self.confusion_img_dir)
            if self.do_validation:
                self._make_a_confusion_matrix(content[f'val_{self.confusion_key}'], 
                                              save_mode=f'Validation {category}', save_dir=self.confusion_img_dir)
            # Save the reuslt of metrics graphs.
            if self.save_performance_plot: plot_performance_N(result, self.metrics_img_dir/f'metrics_graphs_{category}.png')
            close_all_plots()
            
    def _save_tensorboard(self, log):
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
        # Save the value per epoch. And save the value of training and validation.
        if self.tensorboard:
            for category, content in log.items():
                self.writer.set_step(log['epoch'])
                if category in ['epoch', self.confusion_key, f'val_{self.confusion_key}']: continue
                if 'val_' in category or 'time' in category: continue
                if type(content) != dict: # loss
                    if f'val_{category}' in log.keys():
                        scalars = {category:content, f'val_{category}':log[f'val_{category}']}
                        self.writer.add_scalars(category, {str(k):v for k, v in scalars.items()})
                    else: self.writer.add_scalar(category, content)
                elif self.AUCNameForFixedSpec in category: # auc
                    scalars = deepcopy(content)
                    if f'val_{category}' in log.keys(): 
                        scalars.update(**{f'val_{tag}':auc for tag, auc in log[f'val_{category}'].items()})
                    self.writer.add_scalars(category, {str(k):v for k, v in scalars.items()})
                else: # maxprob, Fixed_spec_goal
                    # 1. All metrics (without loss) 2. All metrics per class
                    self._log_metrics_to_tensorboard(content)
            
                    # 3. Confusion Matrix
                    self.writer.set_step(log['epoch'], f'train_{category}', False)
                    self.writer.add_figure(self.confusion_tag_for_writer, self._make_a_confusion_matrix(content[self.confusion_key])) 
                    if f'val_{self.confusion_key}' in content.keys():
                        self.writer.set_step(log['epoch'], f'valid_{category}', False)
                        self.writer.add_figure(self.confusion_tag_for_writer, self._make_a_confusion_matrix(content[f'val_{self.confusion_key}'])) 
                    close_all_plots()