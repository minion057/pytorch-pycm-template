import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTrainer, MetricTracker, ConfusionTracker
from utils import inf_loop, tb_projector_resize, plot_classes_preds, plot_close
import numpy as np

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device)
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.train_confusion = ConfusionTracker(*['confusion'], writer=self.writer, classes=self.classes)
        self.valid_confusion = ConfusionTracker(*['confusion'], writer=self.writer, classes=self.classes)

        self.softmax = nn.Softmax(dim=0)
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None
        
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch. Start with 1.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.train_confusion.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        for batch_idx, (data, target) in enumerate(self.data_loader):
            batch_num = (epoch - 1) * self.len_epoch + batch_idx
                
            # 1. To move Torch to the GPU or CPU
            data, target = data.to(self.device), target.to(self.device)

            # Compute prediction error
            # 2. Clear the gradients of all optimized variables 
            self.optimizer.zero_grad()
            # 3. Forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            logit, predict = torch.max(output, 1)
            if self.loss_fn_name != 'bce_loss': loss = self.criterion(output, target)
            else: loss =  self.criterion(logit, target.type(torch.DoubleTensor).to(self.device))
                
            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # 5. Perform a single optimization step (parameter update)
            self.optimizer.step()

            # 6. Update the loss
            self.writer.set_step(batch_num, 'batch_train')
            self.train_metrics.update('loss', loss.item())
            
            # 7. Update the confusion matrix 
            confusion_content = {'actual':target.cpu().tolist(), 'predict':predict.cpu().tolist()}
            if self.curve_metric_ftns is not None: confusion_content['probability']=[self.softmax(el).tolist() for el in output.detach().cpu()]
            self.train_confusion.update('confusion', confusion_content, img_update=False)

            confusion_obj = self.train_confusion.get_confusion_obj('confusion')
            for met in self.metric_ftns:# pycm version
                self.train_metrics.update(met.__name__, met(confusion_obj, self.classes))
            
            # 7-1. Update the Projector
            if self.train_projector and epoch == 1:                
                # probs 추가하고 싶으면 metadata_header, zip list 이용해서 수정
                label_img, features = tb_projector_resize(data.detach().cpu().clone(), label_img, features)
                class_labels.extend([str(self.classes[lab]) for lab in target.cpu().tolist()])
                
            # 8. Print the result
            if batch_idx % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx+1 == self.len_epoch and self.tensorboard_pred_plot:
                self.prediction_images, self.prediction_labels = data.cpu()[-5:], [self.classes[lab] for lab in target.cpu().tolist()[-5:]]
                data_channel = self.prediction_images.shape[1]
                preds = np.squeeze(predict[-5:].detach().cpu().numpy())
                use_prob = self.train_confusion.get_probability_vector('confusion')[-5:] if self.curve_metric_ftns is not None \
                           else [self.softmax(el).tolist() for el in output[-5:].detach().cpu()]
                self.prediction_preds = [self.classes[lab] for lab in preds]
                self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]                 
            
            if batch_idx == self.len_epoch:     
                break
        
        # 7-2. Upate the example of predtion and Projector
        self.writer.set_step(epoch-1)
        if self.curve_metric_ftns is not None:
            for met in self.curve_metric_ftns:
                curve_fig = met(self.train_confusion.get_actual_vector('confusion'),
                                self.train_confusion.get_probability_vector('confusion'), self.classes)
                self.writer.add_figure(met.__name__, curve_fig)
                if self.save_performance_plot: curve_fig.savefig(self.output_dir / f'{met.__name__}_training.png', bbox_inches='tight')
        if self.train_projector and epoch == 1: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction',
                                   plot_classes_preds(self.prediction_images, self.prediction_labels, self.prediction_preds, self.prediction_probs,
                                                      one_channel = True if data_channel == 1 else False, return_plot=True))
        plot_close()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None

        log = self.train_metrics.result()        
        log_confusion = self.train_confusion.result()

        if self.do_validation:
            val_log, val_confusion = self._valid_epoch(epoch) # Validation Result
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log_confusion.update(**{'val_'+k : v for k, v in val_confusion.items()})
        
        if self.lr_scheduler is not None:
            self.writer.set_step(epoch-1)
            self.writer.add_scalar('lr_schedule', self.optimizer.param_groups[0]['lr'])
            self.lr_scheduler.step()
        
        log.update(log_confusion)            
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_confusion.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                batch_num = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                
                # 1. To move Torch to the GPU or CPU
                data, target = data.to(self.device), target.to(self.device)

                # Compute prediction error
                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                logit, predict = torch.max(output, 1)
                if self.loss_fn_name != 'bce_loss': loss = self.criterion(output, target)
                else: loss =  self.criterion(logit, target.type(torch.DoubleTensor).to(self.device))
                    
                # 3. Update the loss
                self.writer.set_step(batch_num, 'batch_valid')
                self.valid_metrics.update('loss', loss.item())
                # 4. Update the confusion matrix and input data
                confusion_content = {'actual':target.cpu().tolist(), 'predict':predict.cpu().tolist()}
                if self.curve_metric_ftns is not None: confusion_content['probability']=[self.softmax(el).tolist() for el in output.detach().cpu()]
                self.valid_confusion.update('confusion', confusion_content, img_update=False)

                confusion_obj = self.valid_confusion.get_confusion_obj('confusion')
                for met in self.metric_ftns:# pycm version
                    self.valid_metrics.update(met.__name__, met(confusion_obj, self.classes))
                    
                if batch_idx % self.log_step == 0: self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                # 4-1. Update the Projector
                if self.valid_projector and epoch == 1:                    
                    label_img, features = tb_projector_resize(data.detach().cpu().clone(), label_img, features)
                    class_labels.extend([str(self.classes[lab]) for lab in target.cpu().tolist()])
                
                if batch_idx+1 == len(self.valid_data_loader) and self.tensorboard_pred_plot:
                    self.prediction_images, self.prediction_labels = data.cpu()[-5:], [self.classes[lab] for lab in target.cpu().tolist()[-5:]]
                    data_channel = self.prediction_images.shape[1]
                    preds = np.squeeze(predict[-5:].detach().cpu().numpy())
                    use_prob = self.valid_confusion.get_probability_vector('confusion')[-5:] if self.curve_metric_ftns is not None \
                               else [self.softmax(el).tolist() for el in output[-5:].detach().cpu()]
                    self.prediction_preds = [self.classes[lab] for lab in preds]
                    self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]  
                    
        # 4-2. Upate the example of predtion
        self.writer.set_step(epoch-1, 'valid')
        if self.curve_metric_ftns is not None:
            for met in self.curve_metric_ftns:
                curve_fig = met(self.valid_confusion.get_actual_vector('confusion'),
                                self.valid_confusion.get_probability_vector('confusion'), self.classes)
                self.writer.add_figure(met.__name__, curve_fig)
                if self.save_performance_plot: curve_fig.savefig(self.output_dir / f'{met.__name__}_validation.png', bbox_inches='tight')
        if self.valid_projector and epoch == 1:            
            self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction',
                                   plot_classes_preds(self.prediction_images, self.prediction_labels, self.prediction_preds, self.prediction_probs, 
                                                      one_channel = True if data_channel == 1 else False, return_plot=True))
        plot_close()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None
        
        # 5. Print the result
        confusion_obj = self.valid_confusion.get_confusion_obj('confusion')
        self.logger.debug(f'Valid Epoch: {epoch} [last batch] | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_confusion.result()

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
