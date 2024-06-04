from .trainer import Trainer
from base import FixedSpecConfusionTracker
from copy import deepcopy
from pathlib import Path
from utils import read_json, write_dict2json, plot_confusion_matrix_1, plot_performance_N, close_all_plots
import numpy as np

class FixedSpecTrainer(Trainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device,
                         data_loader, valid_data_loader, lr_scheduler, len_epoch)
        self.config = config
        self.device = device

        # Removing duplicate AUC calculation since the trainer already computes it.
        for met in self.metric_ftns:
            if met.__name__.lower() == 'auc': self.metric_ftns.remove(met)

        curve_metrics = self.config.config['curve_metrics'] if 'curve_metrics' in self.config.config.keys() else  None
        self.FixedNegativeROC, self.original_result_name  = None, 'maxprob'
        self.auc = {class_name:None for class_name in self.classes}
        if curve_metrics is not None:
            if 'FixedNegativeROC' in curve_metrics.keys(): 
                self.FixedNegativeROC ={
                    'goal_score':curve_metrics['FixedNegativeROC']['fixed_goal'],
                    'negative_class_idx':curve_metrics['FixedNegativeROC']['negative_class_idx'],
                    'output_dir': self.output_dir / f"{curve_metrics['FixedNegativeROC']['save_dir']}",
                }
                self.goal_for_auc = self.FixedNegativeROC['goal_score'][0]
                self.FixedNegativeROC['output_metrics'] = self.FixedNegativeROC['output_dir'] / 'metrics.json'
                if not self.FixedNegativeROC['output_dir'].is_dir(): self.FixedNegativeROC['output_dir'].mkdir(parents=True, exist_ok=True)
                self.train_FixedNegativeROC = FixedSpecConfusionTracker(goal_score=self.FixedNegativeROC['goal_score'],  classes=self.classes,
                                                                        negative_class_idx=self.FixedNegativeROC['negative_class_idx'])
                self.valid_FixedNegativeROC = FixedSpecConfusionTracker(goal_score=self.FixedNegativeROC['goal_score'], classes=self.classes,
                                                                        negative_class_idx=self.FixedNegativeROC['negative_class_idx'])
            else: raise ValueError('Warring: FixedNegativeROC is not in the config[curve_metrics]')
        else: print('Warring: curve_metrics is not in the config')
       
    def _curve_metrics(self, mode='training'):
        for met in self.curve_metric_ftns:
            if mode=='training':
                actual_vector = self.train_confusion.get_actual_vector(self.confusion_key)
                probability_vector = self.train_confusion.get_probability_vector(self.confusion_key)
            else:
                actual_vector = self.valid_confusion.get_actual_vector(self.confusion_key)
                probability_vector = self.valid_confusion.get_probability_vector(self.confusion_key)
            if met.__name__ == 'FixedNegativeROC': 
                save_path = self.FixedNegativeROC['output_dir'] / f'{met.__name__}_{mode}.png'
                curve_fig = met(actual_vector, probability_vector, self.classes, self.FixedNegativeROC['negative_class_idx'])
            else: 
                save_path = self.output_dir / f'{met.__name__}_{mode}.png'
                curve_fig = met(actual_vector, probability_vector, self.classes)
            self.writer.add_figure(met.__name__, curve_fig)
            if self.save_performance_plot: curve_fig.savefig(save_path, bbox_inches='tight')
        
    def _get_a_log(self, epoch):
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
        log = self.train_metrics.result()
        log_confusion = self.train_confusion.result()
        if self.do_validation:
            val_log, val_confusion = self._valid_epoch(epoch) # Validation Result
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
        
        # Goal Result (FixedSpec)
        log.update(self._FixedNegativeROCResult(mode='training'))
        if self.do_validation:
            val_log = self._FixedNegativeROCResult(mode='validation')
            for key, value in val_log.items(): log[key].update(**{'val_'+k : v for k, v in value.items()})
        
        # AUC Result
        log['auc'] = {}
        if self.do_validation: log['val_auc'] = {}
        for pos_class_idx, pos_class_name in self.train_FixedNegativeROC.positive_classes.items():
            log['auc'][pos_class_name] = self.train_FixedNegativeROC.get_auc(self.goal_for_auc, pos_class_name)
            if self.do_validation: log['val_auc'][pos_class_name] = self.valid_FixedNegativeROC.get_auc(self.goal_for_auc, pos_class_name)
        
        # model save and reset
        self._save_FixedBestModel(epoch)
        self.train_FixedNegativeROC.reset()
        self.valid_FixedNegativeROC.reset()
        return log
    
    def _FixedNegativeROCResult(self, mode='training'):    
        if mode=='training':
            FixedNegativeROC = self.train_FixedNegativeROC
            self.train_FixedNegativeROC.update(self.train_confusion.get_actual_vector(self.confusion_key),
                                               self.train_confusion.get_probability_vector(self.confusion_key), img_update=False)
        else:
            FixedNegativeROC = self.valid_FixedNegativeROC
            self.valid_FixedNegativeROC.update(self.valid_confusion.get_actual_vector(self.confusion_key),
                                               self.valid_confusion.get_probability_vector(self.confusion_key), img_update=False)
        goal_metrics = {}
        # 1. AUC: Pass
        # 2. Metrics
        for goal, pos_class_name in FixedNegativeROC.index:
            use_key = f'fixedSpec_{goal*100:.0f}_Positive_{pos_class_name}'
            goal_metrics[use_key] = {}
            confusion_obj = FixedNegativeROC.get_confusion_obj(goal, pos_class_name)
            
            for met in self.metric_ftns:# pycm version
                met_name_idx = self.metrics_class_index[met.__name__]
                if met_name_idx is None: goal_metrics[use_key][met.__name__] = met(confusion_obj, self.classes)
                else: goal_metrics[use_key][met.__name__] = met(confusion_obj, self.classes, met_name_idx)
                
            goal_metrics[use_key][self.confusion_key] = confusion_obj.to_array().tolist()
        return goal_metrics
    
    def _save_FixedBestModel(self, epoch): 
        # 모델을 저장할지 여부: 이전보다 auc area가 큰가
        self.logger.info('')
        save_model_name = f'model_best_AUC_Positive'
        message = 'Saving current best AUC model'
        FixedNegativeROC = self.train_FixedNegativeROC if self.do_validation else self.valid_FixedNegativeROC
        for pos_class_idx, pos_class_name in FixedNegativeROC.positive_classes.items():
            auc_area = FixedNegativeROC.get_auc(self.goal_for_auc, pos_class_name)            
            if self.auc[pos_class_name] is None or self.auc[pos_class_name] < auc_area:
                self.auc[pos_class_name] = auc_area
                self._save_checkpoint(epoch, filename=f'{save_model_name}_{pos_class_name}', message=message)
    
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
                plot_confusion_matrix_1(val[self.confusion_key], self.classes, 'Confusion Matrix: Training Data', self.output_dir/'confusion_matrix_training.png')
                if 'val_confusion' in list(val.keys()):
                    plot_confusion_matrix_1(val['val_confusion'], self.classes, 'Confusion Matrix: Validation Data', self.output_dir/'confusion_matrix_validation.png')
                save_metrics_path = self.output_metrics
            else: save_metrics_path = Path(str(self.FixedNegativeROC['output_metrics']).replace('.json', f'_{key}.json'))
            
            # Save the result of metrics.
            if save_metrics_path.is_file():
                result = read_json(save_metrics_path) 
                for k, v in result.items():
                    if k == 'totaltime': continue
                    
                    # Adjusting the number of results per epoch.
                    if 'auc' in k:
                        for pos_class_name, auc_list in v.items():
                            if len(auc_list) != int(basic_log['epoch']): 
                                result[k][pos_class_name]=result[k][pos_class_name][:int(basic_log['epoch'])-1]
                    elif len(v) != int(basic_log['epoch']): result[k]=result[k][:int(basic_log['epoch'])-1]
                    
                    # Adding additional information by default.
                    if k in basic_log.keys(): result[k].append(basic_log[k])
                    elif k in auc_log.keys():
                        for pos_class_name, auc_area in auc_log[k].items(): result[k][pos_class_name].append(auc_area)
                    else: result[k].append(val[k])
            else:
                result = {k:[v] for k, v in basic_log.items()}
                for k, v in auc_log.items():
                    result[k] = {pos_class_name:[auc_area] for pos_class_name, auc_area in v.items()}
                for k, v in val.items(): result[k] = [v]  
            write_dict2json(result, save_metrics_path)

            # Save the reuslt of metrics graphs.
            save_dir = self.output_dir if key == self.original_result_name else self.FixedNegativeROC['output_dir']
            if self.save_performance_plot: plot_performance_N(result, save_dir/f'metrics_graphs_{key}.png')
            close_all_plots()
            
    def _save_tensorboard(self, log):
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
        # Save the value per epoch. And save the value of training and validation.
        if self.tensorboard:
            for key, value in log.items():
                self.writer.set_step(log['epoch']-1)
                if key in ['epoch', 'confusion', 'val_confusion']: continue
                if 'val_' in key or 'time' in key: continue
                if type(value) != dict: # loss
                    if f'val_{key}' in log.keys():
                        content = {key:value, f'val_{key}':log[f'val_{key}']}
                        self.writer.add_scalars(key, {str(k):v for k, v in content.items()})
                    else: self.writer.add_scalar(key, value)
                elif 'auc' in key:
                    content = deepcopy(value)
                    if f'val_{key}' in log.keys(): content.update(**{f'val_{pos_class_name}':auc_area for pos_class_name, auc_area in log[f'val_{key}'].items()})
                    self.writer.add_scalars(key, {str(k):v for k, v in content.items()})
                else: #maxprob, Fixed_spec_goal
                    for new_key, new_value in value.items():
                        if new_key in ['epoch', 'confusion', 'val_confusion']: continue
                        if 'val_' in new_key or 'time' in new_key: continue
                        # 1. All metrics
                        if '_class' not in new_key:
                            if f'val_{new_key}' in value.keys():
                                content = {new_key:new_value, f'val_{new_key}':value[f'val_{new_key}']}
                                self.writer.add_scalars(new_key, {str(k):v for k, v in content.items()})
                            else: self.writer.add_scalar(new_key, new_value)
                        # 2. All metrics per class
                        else:
                            if f'val_{new_key}' in value.keys():
                                content = deepcopy(new_value)
                                content.update({f'val_{k}':v for k, v in value[f'val_{new_key}'].items()})
                                self.writer.add_scalars(new_key, {str(k):v for k, v in content.items()})
                            else: self.writer.add_scalars(new_key, {str(k):v for k, v in new_value.items()})
            
                    # 3. Confusion Matrix
                    self.writer.set_step(log['epoch']-1, f'train_{key}')
                    tag = f'ConfusionMatrix'
                    self.writer.add_figure(tag, plot_confusion_matrix_1(value['confusion'], self.classes, return_plot=True))
                    if 'val_confusion' in value.keys():
                        self.writer.set_step(log['epoch']-1, f'valid_{key}')
                        self.writer.add_figure(tag, plot_confusion_matrix_1(value['val_confusion'], self.classes, return_plot=True))
                    close_all_plots()