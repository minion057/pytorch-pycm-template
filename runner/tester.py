import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTester, MetricTracker, ConfusionTracker
from utils import tb_projector_resize, plot_classes_preds, close_all_plots, save_pycm_object, check_onehot_encoding_1
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

class Tester(BaseTester):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader, is_test:bool=True):
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, is_test)
        self.config = config
        self.device = device
        
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.wirter_mode = f'test'

        metrics_tag = []
        for met in self.metric_ftns:
            met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
            metrics_tag.append(met.__name__ if tag is None else tag)
        self.basic_metrics = ['loss']
        self.metrics = MetricTracker(*self.basic_metrics, *metrics_tag, writer=self.writer)
        self.confusion = ConfusionTracker(*[self.confusion_key], writer=self.writer, classes=self.classes)

        self.softmax = nn.Softmax(dim=0)
        self.preds_item_cnt = 5
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None

    def _test(self):
        """
        Test logic
        """
        self.model.eval()
        self.metrics.reset()
        self.confusion.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
                batch_num = batch_idx + 1
                
                # 1. To move Torch to the GPU or CPU
                data, target = data.to(self.device), target.to(self.device)

                # Compute prediction error
                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                logit, predict = torch.max(output, 1)
                loss = self._loss(output, target, logit)
                if check_onehot_encoding_1(target[0].cpu(), self.classes): target = torch.max(target, 1)[-1] # indices
                
                use_data, use_target = data.detach().cpu(), target.detach().cpu().tolist()
                use_output, use_predict =output.detach().cpu(), predict.detach().cpu()
                
                # 3. Update the loss
                self.writer.set_step(batch_num, f'batch_{self.wirter_mode}')
                self.metrics.update('loss', loss.item())
                # 4. Update the confusion matrix and input data
                confusion_content = {'actual':use_target, 'predict':use_predict.clone().tolist(), 'probability':[self.softmax(el).tolist() for el in use_output]}
                self.confusion.update(self.confusion_key, confusion_content, img_update=False)

                confusion_obj = self.confusion.get_confusion_obj(self.confusion_key)
                for met in self.metric_ftns:
                    met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]), met_name=met.__name__)
                    use_confusion_obj = deepcopy(confusion_obj)                             
                    if met_kwargs is None: self.metrics.update(tag, met(use_confusion_obj, self.classes))
                    else: self.metrics.update(tag, met(use_confusion_obj, self.classes, **met_kwargs))
                    
                if batch_idx % self.log_step == 0:
                    self.writer.add_image('input', make_grid(use_data, nrow=8, normalize=True))
                
                # 4-1. Update the Projector
                if self.projector:                    
                    label_img, features = tb_projector_resize(use_data.clone(), label_img, features)
                    class_labels.extend([str(self.classes[lab]) for lab in use_target])
                
                if batch_idx == len(self.data_loader)-2 and self.tensorboard_pred_plot:
                    # last batch -1 > To minimize batches with a length of 1 as much as possible.
                    # If you want to modify the last batch, pretend that len(self.data_loader)-2 is self.len_epoch-1.
                    self.prediction_images, self.prediction_labels = use_data[-self.preds_item_cnt:], [self.classes[lab] for lab in use_target[-self.preds_item_cnt:]]
                    data_channel = self.prediction_images.shape[1]
                    preds = np.squeeze(use_predict[-self.preds_item_cnt:].numpy())                    
                    preds = preds if len(target)!=1 else np.array([preds]) # For batches with length of 1                 
                    use_prob = self.confusion.get_probability_vector(self.confusion_key)[-len(preds):] 
                    self.prediction_preds = [self.classes[lab] for lab in preds]
                    self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]  
        
        # 4-2. Update the curve plot and projector
        self.writer.set_step(self.test_epoch, self.wirter_mode)
        if self.plottable_metric_ftns is not None: self._plottable_metrics()
        if self.projector: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        # 4-3. Upate the example of predtion
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction', plot_classes_preds(self.prediction_images, self.prediction_labels, 
                                                                    self.prediction_preds, self.prediction_probs,
                                                                    one_channel = True if data_channel == 1 else False, return_plot=True))
        close_all_plots()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None
        
        # 5. setting result
        return self._get_a_log()

    def _loss(self, output, target, logit):
        if self.loss_fn_name != 'bce_loss': loss = self.criterion(output, target)
        else: loss =  self.criterion(logit, target.type(torch.DoubleTensor).to(self.device))
        return loss
    
    def _plottable_metrics(self):
        actual_vector = self.confusion.get_actual_vector(self.confusion_key)
        probability_vector = self.confusion.get_probability_vector(self.confusion_key)
        for met in self.plottable_metric_ftns:
            met_kwargs, tag, save_dir = self._set_metric_kwargs(deepcopy(self.plottable_metrics_kwargs[met.__name__]), met_name=met.__name__)
            save_dir = self.output_dir / 'plottable_metrics' if save_dir is None else self.output_dir / save_dir
            if met_kwargs is None: fig = met(actual_vector, probability_vector, self.classes)
            else: fig = met(actual_vector, probability_vector, self.classes, **met_kwargs)
            self.writer.add_figure(tag, fig)
            if self.save_performance_plot: 
                if not save_dir.is_dir(): save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f'{tag}_test.png', bbox_inches='tight') # -epoch{self.test_epoch}
    
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
    
    def _get_a_log(self):
        log = self.metrics.result()
        log_confusion = self.confusion.result()
        log.update(log_confusion) 
        return log
    
    def _save_other_output(self, log):
        self._save_confusion_obj()
        
    def _save_confusion_obj(self, filename='cm', message='Saving checkpoint for Confusion Matrix'):
        save_pycm_object(self.confusion.get_confusion_obj(self.confusion_key), save_dir=self.output_dir, save_name= filename)