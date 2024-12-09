import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTrainer, MetricTracker, ConfusionTracker, BaseTracker
from utils import ensure_dir, inf_loop, register_forward_hook_layer, tb_projector_resize, check_onehot_encoding_1
from utils import plot_classes_preds, close_all_plots, save_pycm_object
import numpy as np
from copy import deepcopy
from inspect import signature
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device, data_loader, 
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, da_ftns=None):
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device)
        self.logger.info('Trainer is running...')
        self.config = config
        self.device = device
        
        if len_epoch is None:
            self.data_loader = data_loader
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.metric_loss_key = 'loss'
        self.basic_metrics = [self.metric_loss_key]
        metrics_tag = []
        for met in self.metric_ftns:
            met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
            metrics_tag.append(met.__name__ if tag is None else tag)
        self.train_metrics = MetricTracker(*self.basic_metrics, *metrics_tag, writer=self.writer)
        self.valid_metrics = MetricTracker(*self.basic_metrics, *metrics_tag, writer=self.writer)

        self.train_confusion = ConfusionTracker(*[self.confusion_key], writer=self.writer, classes=self.classes)
        self.valid_confusion = ConfusionTracker(*[self.confusion_key], writer=self.writer, classes=self.classes)
        self.confusion_obj_dir = self.output_dir / 'confusion_object'
        ensure_dir(self.confusion_obj_dir)
        
        # Additional Tracking
        self.additionalTracking_key, self.additionalTracking_columns = 'meta', ['path', 'target', 'pred']
        self.additionalTracking_columns.extend([f'prob:{c}' for c in self.classes])
        self.train_additionalTracking = BaseTracker(*[self.additionalTracking_key], columns=self.additionalTracking_columns)
        self.valid_additionalTracking = BaseTracker(*[self.additionalTracking_key], columns=self.additionalTracking_columns)
        self.additional_dir = self.output_dir / 'metadata'
        ensure_dir(self.additional_dir)

        self.softmax = nn.Softmax(dim=0)
        self.preds_item_cnt = 5
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None

        # Hook for DA
        # if self.cfg_da is not None: self.DA_ftns = getattr(module_DA, self.DA)(writer=self.writer, **self.DAargs)  
        self.DA_ftns = None
        if da_ftns is not None:
            self.hookargs = config['data_augmentation']['hook_args']
            self.DA_ftns = da_ftns(writer=self.writer, **config['data_augmentation']['args'])  
        
        # Clear the gradients of all optimized variables 
        self.optimizer.zero_grad()
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch. Start with 1.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.train_confusion.reset()
        self.train_additionalTracking.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        
        # # Perform DA with hooking
        if self.DA_ftns is not None and self.hookargs is not None:
            hook = register_forward_hook_layer(self.model, 
                                                self.DA_ftns.forward_pre_hook if self.hookargs['pre']else self.DA_ftns.forward_hook, 
                                                **self.hookargs)
                
        for batch_idx, load_data in enumerate(self.data_loader):
            if len(load_data) == 3: 
                data, target, path = load_data
                path = list(path)
            elif len(load_data) == 2: data, target, path = load_data, None
            else: raise Exception('The length of load_data should be 2 or 3.')
            
            batch_num = (epoch - 1) * self.len_epoch + batch_idx + 1
            self.writer.set_step(batch_num, 'batch_train')
                
            # 1. To move Torch to the GPU or CPU
            if self.DA_ftns is not None and self.hookargs is None: # Perform DA without hooking
                data = self.DA_ftns.without_hook(data)
            # if batch_idx == 0: target[target == 2] = 1
            # target_classes, target_cnt = np.unique(target, axis=0, return_counts=True)
            # print(f'target classes: {target_classes.tolist()}, target cnt: {target_cnt.tolist()}')
            data, target = data.to(self.device), target.to(self.device)

            # Compute prediction error
            # 2. Forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            logit, predict = torch.max(output, 1)
            # DA will be replaced randomly if prob is given, i.e. not incremented. To increment, set no probability or use a sampler.
            if self.DA_ftns is None: loss = self._loss(output, target, logit)
            else: 
                if path is not None:
                    path.extend([f'{self.DA_ftns.type}_{p}' for p in np.array(path)[self.DA_ftns.rand_index()]])
                loss, target = self._da_loss(output, target, logit)
            if check_onehot_encoding_1(target[0].cpu(), self.classes): target = torch.max(target, 1)[-1] # indices
            
            # 3. Backward pass: compute gradient of the loss with respect to model parameters
            if self.accumulation_steps is not None: loss = loss / self.accumulation_steps
            loss.backward()
            # 4-1. Perform a single optimization step (parameter update)
            # 4-2. Clear the gradients of all optimized variables 
            if self.accumulation_steps is not None:
                if batch_idx == self.len_epoch-1 or self.len_epoch == 1 or (batch_idx+1) % self.accumulation_steps == 0: 
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else: 
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # 5. Update the Result
            use_data, use_target = data.detach().cpu(), target.detach().cpu().tolist()
            use_output, use_predict =output.detach().cpu(), predict.detach().cpu()
            # 5-1. loss
            self.train_metrics.update('loss', loss.item())
            
            # 5-2. confusion matrix 
            confusion_content = {'actual':use_target, 'predict':use_predict.tolist(), 'probability':[self.softmax(el).tolist() for el in use_output]}
            self.train_confusion.update(self.confusion_key, confusion_content, img_update=False)
            
            confusion_obj = self.train_confusion.get_confusion_obj(self.confusion_key)
            for met in self.metric_ftns:# pycm version
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
                use_confusion_obj = deepcopy(confusion_obj)   
                met_result = met(use_confusion_obj, self.classes) if met_kwargs is None else met(use_confusion_obj, self.classes, **met_kwargs)
                self.train_metrics.update(tag, met_result)
            
            # 5-3. Additional tracking
            for col in self.additionalTracking_columns:
                if 'path' in col.lower() and path is not None: 
                    for p in path: self.train_additionalTracking.update(self.additionalTracking_key, col, str(p))
                elif 'target' in col.lower():
                    for p in confusion_content['actual']: self.train_additionalTracking.update(self.additionalTracking_key, col, p)
                elif 'pred' in col.lower():
                    for p in confusion_content['predict']: self.train_additionalTracking.update(self.additionalTracking_key, col, self.classes[p])
                elif 'prob' in col.lower():
                    class_idx = [i for i, c in enumerate(self.classes) if c in col]
                    if len(class_idx) != 1: 
                        raise ValueError(f'All class names could not be found in the name of the column ({col}) where the probability value is to be stored.')
                    for p in confusion_content['probability']: 
                        self.train_additionalTracking.update(self.additionalTracking_key, col, p[class_idx[0]])
            
            # 5-3-1. Projector
            # The data concerning the projector is collected with each batch and will be updated after all batches are completed.
            # See 5-3-2.
            if self.train_projector and epoch == 1:                
                label_img, features = tb_projector_resize(use_data.clone(), label_img, features)
                class_labels.extend([str(self.classes[lab]) for lab in use_target])
                
            # 6. Print the result
            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch-1 or self.len_epoch == 1:
                if confusion_obj.Overall_ACC=='None':
                    print(f'Now confusion matrix is\m{confusion_obj.print_matrix()}')
                    raise ValueError('Confusion matrix calculation failed. Please check the input data.')
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
                self.writer.add_image('input', make_grid(use_data, nrow=8, normalize=True))
            
            if (batch_idx == self.len_epoch-2 or self.len_epoch == 1) and self.tensorboard_pred_plot: 
                # last batch -1 > To minimize batches with a length of 1 as much as possible.
                # If you want to modify the last batch, pretend that self.len_epoch-2 is self.len_epoch-1.
                self.prediction_images, self.prediction_labels = use_data[-self.preds_item_cnt:], [self.classes[lab] for lab in use_target[-self.preds_item_cnt:]]
                data_channel = self.prediction_images.shape[1]
                preds = np.squeeze(use_predict[-self.preds_item_cnt:].numpy())
                preds = preds if len(target)!=1 else np.array([preds]) # For batches with length of 1                 
                use_prob = self.train_confusion.get_probability_vector(self.confusion_key)[-len(preds):] 
                self.prediction_preds = [self.classes[lab] for lab in preds]
                self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]          
            if batch_idx == self.len_epoch: break
        
        if self.DA_ftns is not None and self.hookargs is not None: hook.remove()
        # 5-3-2. Update the curve plot and projector
        self.writer.set_step(epoch)
        if self.plottable_metric_ftns is not None: 
            self._plottable_metrics(actual_vector = self.train_confusion.get_actual_vector(self.confusion_key),
                                    probability_vector = self.train_confusion.get_probability_vector(self.confusion_key),
                                    mode='training')
        if self.train_projector and epoch == 1: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        # 5-3-3. Upate the example of predtion
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction', plot_classes_preds(self.prediction_images, self.prediction_labels, 
                                                                    self.prediction_preds, self.prediction_probs,
                                                                    one_channel = True if data_channel == 1 else False, return_plot=True))
        close_all_plots()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None   
        
        # 6. Upate the lr scheduler
        if self.lr_scheduler is not None and not self.do_validation:
            self.writer.set_step(epoch)
            self._lr_scheduler.step(epoch=epoch, mode='train')

        # 7. setting result     
        return self._get_a_log(epoch)
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_confusion.reset()
        self.valid_additionalTracking.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        # print('\n\n\nValid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.writer.set_step(epoch, 'valid')
        with torch.no_grad():
            output, use_data, use_target, use_path = [], [], [], []
            for batch_idx, load_data in enumerate(self.valid_data_loader):
                if len(load_data) == 3: data, target, path = load_data
                elif len(load_data) == 2: data, target, path = load_data, None
                else: raise Exception('The length of load_data should be 2 or 3.')
                
                
                # 1. To move Torch to the GPU or CPU
                # target_classes, target_cnt = np.unique(target, axis=0, return_counts=True)
                # print(f'target classes: {target_classes.tolist()}, target cnt: {target_cnt.tolist()}')
                data, target = data.to(self.device), target.to(self.device)

                # Compute prediction error
                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output.append(self.model(data).detach())
                use_data.append(data.detach())
                use_target.append(target.detach())
                use_path.extend(list(path))
            
            output = torch.cat(output, 0)
            use_data = torch.cat(use_data, 0)
            use_target = torch.cat(use_target, 0)
            logit, predict = torch.max(output, 1) 
            loss = self._loss(output, use_target, logit)
            if check_onehot_encoding_1(use_target[0].cpu(), self.classes): use_target = torch.max(use_target, 1)[-1] # indices
            
            use_data, use_target = use_data.cpu(), use_target.cpu().tolist()
            use_output, use_predict = output.cpu(), predict.detach().cpu()
            
            # 3. Update the loss
            self.valid_metrics.update('loss', loss.item())
            # 4. Update the confusion matrix and input data
            confusion_content = {'actual':use_target, 'predict':use_predict.tolist(), 'probability':[self.softmax(el).tolist() for el in use_output]}
            self.valid_confusion.update(self.confusion_key, confusion_content, img_update=False)
                
            confusion_obj = self.valid_confusion.get_confusion_obj(self.confusion_key)
            for met in self.metric_ftns:# pycm version
                met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
                use_confusion_obj = deepcopy(confusion_obj)          
                if met_kwargs is None: self.valid_metrics.update(tag, met(use_confusion_obj, self.classes))
                else: self.valid_metrics.update(tag, met(use_confusion_obj, self.classes, **met_kwargs))               
            self.writer.add_image('input', make_grid(use_data, nrow=8, normalize=True))
            
            # 4-0. Additional tracking
            for col in self.additionalTracking_columns:
                if 'path' in col.lower() and path is not None: 
                    for p in path: self.valid_additionalTracking.update(self.additionalTracking_key, col, str(p))
                elif 'target' in col.lower():
                    for p in confusion_content['actual']: self.valid_additionalTracking.update(self.additionalTracking_key, col, p)
                elif 'pred' in col.lower():
                    for p in confusion_content['predict']: self.valid_additionalTracking.update(self.additionalTracking_key, col, self.classes[p])
                elif 'prob' in col.lower():
                    class_idx = [i for i, c in enumerate(self.classes) if c in col]
                    if len(class_idx) != 1: 
                        raise ValueError(f'All class names could not be found in the name of the column ({col}) where the probability value is to be stored.')
                    for p in confusion_content['probability']: 
                        self.valid_additionalTracking.update(self.additionalTracking_key, col, p[class_idx[0]])
        
            # 4-1. Update the Projector
            if self.valid_projector and epoch == 1:                    
                label_img, features = tb_projector_resize(use_data.clone(), label_img, features)
                class_labels.extend([str(self.classes[lab]) for lab in use_target])
            
            if self.tensorboard_pred_plot:
                self.prediction_images, self.prediction_labels = use_data[-self.preds_item_cnt:], [self.classes[lab] for lab in use_target[-self.preds_item_cnt:]]
                data_channel = self.prediction_images.shape[1]
                preds = np.squeeze(use_predict[-self.preds_item_cnt:].numpy())
                preds = preds if len(use_target)!=1 else np.array([preds]) # For batches with length of 1  
                use_prob = self.valid_confusion.get_probability_vector(self.confusion_key)[-len(preds):] 
                self.prediction_preds = [self.classes[lab] for lab in preds]
                self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]  
                    
        # 4-2. Update the curve plot and projector
        if self.plottable_metric_ftns is not None:
            self._plottable_metrics(actual_vector = self.valid_confusion.get_actual_vector(self.confusion_key),
                                    probability_vector = self.valid_confusion.get_probability_vector(self.confusion_key),
                                    mode='validation')
        if self.valid_projector and epoch == 1: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        # 4-3. Upate the example of predtion
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction', plot_classes_preds(self.prediction_images, self.prediction_labels, 
                                                                    self.prediction_preds, self.prediction_probs,
                                                                    one_channel = True if data_channel == 1 else False, return_plot=True))
        close_all_plots()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None   
        
        # 5. Print the result
        confusion_obj = self.valid_confusion.get_confusion_obj(self.confusion_key)
        if confusion_obj.Overall_ACC=='None':
            print(f'Now confusion matrix is\m{confusion_obj.print_matrix()}')
            raise ValueError('Confusion matrix calculation failed. Please check the input data.')
        self.logger.debug(f'Valid Epoch: {epoch} | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
        
        # 6. add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        
        # 7. Upate the lr scheduler
        if self.lr_scheduler is not None:
            self.writer.set_step(epoch, 'valid')
            self._lr_scheduler(epoch=epoch, mode='valid')

    def _loss(self, output, target, logit):
        return self.criterion(output, target, self.classes, self.device)
        
    def _da_loss(self, output, target, logit):     
        if 'loss' not in dir(self.DA_ftns):
            raise AttributeError('Loss function is not defined in the DA class. If you need a specific formula, configure a loss function.')
        result = self.DA_ftns.loss(self._loss, output, target, logit)
        if not isinstance(result, dict) or result=={} or not all(k in ['loss', 'target'] for k in result.keys()): 
            raise ValueError('The loss function in the DA class requires passing values as a dictionary with keys "loss" and "target".')
        self.DA_ftns.reset()
        return result['loss'], result['target']
    
    def _lr_scheduler(self, epoch, mode='training'):
        if mode.lower() not in ['training', 'validation', 'train', 'valid']: 
            raise ValueError('mode must be "training" ("train") or "validation" ("valid").')
        self.writer.add_scalar('lr_schedule', self.optimizer.param_groups[0]['lr'])
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): 
            if mode == 'training':
                self.logger.warning('ReduceLROnPlateau requires a validation metric (e.g., validation loss) to function properly.\n'
                                    +'Using training loss instead may lead to overfitting. Therefore, the learning rate scheduler does not apply.\n'
                                    +'Please ensure a validation dataset and metric are provided.')
                return
            self.lr_scheduler.step(self.valid_metrics.avg(self.metric_loss_key))
        else: 
            try: self.lr_scheduler.step()
            except: self.lr_scheduler.step(epoch) # for timm optim
     
    def _plottable_metrics(self, actual_vector, probability_vector, mode='training'):
        if np.array(probability_vector).ndim != 2: raise ValueError('The probability vector should be 2D array.')
        for met in self.plottable_metric_ftns:            
            met_kwargs, tag, save_dir = self._set_metric_kwargs(deepcopy(self.plottable_metrics_kwargs[met.__name__]), met_name=met.__name__)
            save_dir = self.output_dir / 'plottable_metrics' if save_dir is None else self.output_dir / save_dir
            if met_kwargs is None: fig = met(actual_vector, probability_vector, self.classes)
            else: fig = met(actual_vector, probability_vector, self.classes, **met_kwargs)
            self.writer.add_figure(tag, fig)
            if self.save_performance_plot: 
                if not save_dir.is_dir(): ensure_dir(save_dir, True)
                fig.savefig(save_dir / f'{tag}_{mode}.png', bbox_inches='tight')
        
    def _set_metric_kwargs(self, met_kwargs, met_name:str=None):
        if met_kwargs is None: return None, None, None
        if 'tag' in met_kwargs: 
            tag = met_kwargs['tag']
            met_kwargs.pop('tag')
        else: 
            if met_name is None: raise ValueError("Expected 'met_name' to be not None, but received None.")
            tag = met_name
        if 'save_dir' in met_kwargs: 
            save_dir = met_kwargs['save_dir']
            met_kwargs.pop('save_dir') 
        else: save_dir = None
        return met_kwargs, tag, save_dir
        
    def _get_a_log(self, epoch):        
        log, log_confusion = self.train_metrics.result(), self.train_confusion.result()
        
        if self.do_validation:
            self._valid_epoch(epoch)
            val_log, val_confusion = self.valid_metrics.result(),  self.valid_confusion.result()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log_confusion.update(**{'val_'+k : v for k, v in val_confusion.items()})
        log.update(log_confusion)    
        log = self._sort_train_val_sequences(log)       
        return log
            
    def _progress(self, batch_idx):
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        
        str_diff = len(str(total))-len(str(current))
        current_str = str(current) if str_diff == 0 else ' '*str_diff+str(current)
        percentage = f'{100.0 * (current/total):.0f}' 
        return f'[{current_str}/{total} ({percentage:2s})%]'
    
    def _save_other_output(self, epoch, log, save_best:bool=False):
        last_filename, best_filename = '_latest', '_model_best'
        
        self._save_confusion_obj(filename=f'cm{last_filename}')
        self._save_path_infomation(filename=f'result{last_filename}')
        if save_best: 
            self._save_confusion_obj(filename=f'cm{best_filename}')
            self._save_path_infomation(filename=f'result{best_filename}')
        
    def _save_confusion_obj(self, filename:str, message='Saving checkpoint for Confusion Matrix'):
        save_pycm_object(self.train_confusion.get_confusion_obj(self.confusion_key), 
                         save_dir=self.confusion_obj_dir, save_name= filename+'_training')
        if self.do_validation: 
            save_pycm_object(self.valid_confusion.get_confusion_obj(self.confusion_key), 
                             save_dir=self.confusion_obj_dir, save_name= filename+'_validation')
            
    def _save_path_infomation(self, filename:str):
        self.train_additionalTracking.save2excel(savedir=self.additional_dir, savename=filename+'_training', excel_type='xlsx')
        if self.do_validation: 
            self.valid_additionalTracking.save2excel(savedir=self.additional_dir, savename=filename+'_validation', excel_type='xlsx')