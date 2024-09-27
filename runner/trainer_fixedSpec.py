import numpy as np
from .trainer import Trainer
from base import FixedSpecConfusionTracker
from copy import deepcopy
from pathlib import Path
from utils import ensure_dir, read_json, write_dict2json
from utils import plot_confusion_matrix_1, plot_performance_N, close_all_plots

class FixedSpecTrainer(Trainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, da_ftns=None):
        # Removing duplicate AUC calculation since the trainer already computes it.
        if 'fixed_goal' not in config['trainer'].keys():
            raise ValueError('There is no fixed specificity score to track.')
        self.ROCNameForFixedSpec, self.AUCNameForFixedSpec = 'ROC_OvO', 'AUC_OvO'
        _metric_ftns = deepcopy(metric_ftns)
        for met in metric_ftns:
            if self.AUCNameForFixedSpec.lower() in met.__name__.lower(): 
                _metric_ftns.remove(met)
                del config['metrics'][met.__name__]
        metric_ftns = _metric_ftns
        
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device,
                         data_loader, valid_data_loader, lr_scheduler, len_epoch, da_ftns)
        self.config = config
        self.device = device
        
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
        basic_log, basic_confusion = self.train_metrics.result(), self.train_confusion.result()
        if self.do_validation:
            self._valid_epoch(epoch)
            val_basic_log, val_basic_confusion = self.valid_metrics.result(),  self.valid_confusion.result()
            basic_log.update(**{'val_'+k : v for k, v in val_basic_log.items()})
            basic_confusion.update(**{'val_'+k : v for k, v in val_basic_confusion.items()})
        
        # Original Result (MaxProb)
        log, original_log = {}, {}
        for k, v in basic_log.items():
            if any(basic in k for basic in self.basic_metrics): log[k] = v
            else: original_log[k] = v
        original_log.update(basic_confusion)
        log = self._sort_train_val_sequences(log)
        log[self.original_result_name] = self._sort_train_val_sequences(original_log)
        
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
        
        # 1. AUCs that exist among the list of metrics
        # AUC already includes 1, so there is only 1 left to track in the metric. 
        # Because AUC only supports OVO and OVR methods. Therefore, it uses the index, not the list of indexes.
        auc_ftns_index = next((i for i, met in enumerate(self.metric_ftns) if 'auc' in met.__name__.lower()), None)
        auc_score, auc_tag = None, None
        if auc_ftns_index is not None:
            met_name = self.metric_ftns[auc_ftns_index].__name__
            met_kwargs, auc_tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met_name]), met_name=met_name)
            if met_kwargs is None: auc_score = self.metric_ftns[auc_ftns_index](maxprob_confusion, self.classes)
            else: auc_score = self.metric_ftns[auc_ftns_index](maxprob_confusion, self.classes, **met_kwargs) 
        
        # 2. Other metrics
        confusion_dict = ROCForFixedSpec.result()
        for goal, pos_class_name, neg_class_name in ROCForFixedSpec.index:
            pos_class_idx, neg_class_idx = np.where(np.array(self.classes) == pos_class_name)[0][0], np.where(np.array(self.classes) == neg_class_name)[0][0]
            category = ROCForFixedSpec.get_tag(goal, pos_class_name, neg_class_name)
            confusion_obj = ROCForFixedSpec.get_confusion_obj(goal, pos_class_name, neg_class_name)
            confusion_classes = np.array(confusion_obj.classes)
            goal_metrics[category] = {}
            for met_idx, met in enumerate(self.metric_ftns): # pycm version
                if met_idx == auc_ftns_index: 
                    goal_metrics[category][auc_tag] = auc_score
                    continue
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
                use_confusion_obj = deepcopy(confusion_obj)
                if met_kwargs is None: goal_metrics[category][tag] = met(use_confusion_obj, self.classes)
                else:
                    # 일반 메트릭 auc, 점수 이상함 확인 필요
                    # 메트릭 플롯에서 딕셔너리면 key+key해서 리스트로 넣어주기
                    print(f'Using "{met.__name__}" for {tag} in {category}')
                    print(f'pos index: {pos_class_idx}, neg index: {neg_class_idx}')
                    print(f'ori met_kwargs: {met_kwargs}')
                    run = True
                    for key, value in met_kwargs.items():
                        if 'idx' in key:
                            if value not in [pos_class_idx, neg_class_idx]: 
                                run = False
                                break
                        elif 'indices' in key: 
                            valid_indices = [index for index in value if index in [pos_class_idx, neg_class_idx]]
                            print('valid_indices: ', valid_indices)
                            if valid_indices == []:
                                run = False
                                break
                            else: met_kwargs[key] = valid_indices
                    if not run: continue
                    print(f'met_kwargs: {met_kwargs}\n')
                    goal_metrics[category][tag] = met(use_confusion_obj, self.classes, **met_kwargs)             
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
        basic_log = {key:val for key, val in log.items() if not isinstance(val, dict)} # epoch, loss, val_loss, runtime
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