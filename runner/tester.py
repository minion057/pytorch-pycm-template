import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTester, MetricTracker, ConfusionTracker
from utils import tb_projector_resize, plot_classes_preds, close_all_plots
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

class Tester(BaseTester):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device, data_loader):
        super().__init__(model, criterion, metric_ftns, plottable_metric_ftns, config, classes, device)
        self.config = config
        self.device = device
        
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.wirter_mode = f'test'

        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.confusion_key = 'confusion'
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
                batch_num = batch_idx
                
                # 1. To move Torch to the GPU or CPU
                data, target = data.to(self.device), target.to(self.device)

                # Compute prediction error
                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                logit, predict = torch.max(output, 1)
                if self.loss_fn_name != 'bce_loss': loss = self.criterion(output, target)
                else: loss =  self.criterion(logit, target.type(torch.DoubleTensor).to(self.device))
                    
                # 3. Update the loss
                self.writer.set_step(batch_num, f'batch_{self.wirter_mode}')
                self.metrics.update('loss', loss.item())
                # 4. Update the confusion matrix and input data
                confusion_content = {'actual':target.cpu().tolist(), 'predict':predict.cpu().tolist()}
                if self.plottable_metric_ftns is not None: confusion_content['probability']=[self.softmax(el).tolist() for el in output.detach().cpu()]
                self.confusion.update(self.confusion_key, confusion_content, img_update=False)

                confusion_obj = self.confusion.get_confusion_obj(self.confusion_key)
                for met in self.metric_ftns:
                    met_kwargs, tag, _ = self._set_metric_kwargs(deepcopy(self.metrics_kwargs[met.__name__]))
                    tag = met.__name__ if tag is None else tag
                    use_confusion_obj = deepcopy(confusion_obj)                             
                    if met_kwargs is None: self.metrics.update(tag, met(use_confusion_obj, self.classes))
                    else: self.metrics.update(tag, met(use_confusion_obj, self.classes, **met_kwargs))
                    
                if batch_idx % self.log_step == 0:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                # 4-1. Update the Projector
                if self.projector:                    
                    label_img, features = tb_projector_resize(data.detach().cpu().clone(), label_img, features)
                    class_labels.extend([str(self.classes[lab]) for lab in target.cpu().tolist()])
                
                if batch_idx == len(self.data_loader)-2 and self.tensorboard_pred_plot:
                    # last batch -1 > To minimize batches with a length of 1 as much as possible.
                    # If you want to modify the last batch, pretend that len(self.data_loader)-2 is self.len_epoch-1.
                    self.prediction_images, self.prediction_labels = data.cpu()[-self.preds_item_cnt:], [self.classes[lab] for lab in target.cpu().tolist()[-self.preds_item_cnt:]]
                    data_channel = self.prediction_images.shape[1]
                    preds = np.squeeze(predict[-self.preds_item_cnt:].detach().cpu().numpy())                    
                    preds = preds if len(target)!=1 else np.array([preds]) # For batches with length of 1                 
                    if self.plottable_metric_ftns is not None:
                        use_prob = self.confusion.get_probability_vector(self.confusion_key)[-len(preds):] 
                    else: use_prob = [self.softmax(el).tolist() for el in output[-len(preds):].detach().cpu()]
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
    
    def _set_metric_kwargs(self, met_kwargs):
        if 'tag' in met_kwargs: 
            tag = met_kwargs['tag']
            met_kwargs.pop('tag')  
        else: tag = None
        if 'save_dir':
            save_dir = met_kwargs['save_dir']
            met_kwargs.pop('save_dir') 
        else: save_dir = None
        return met_kwargs, tag, save_dir
    
    def _plottable_metrics(self):
        for met in self.plottable_metric_ftns:
            met_kwargs, tag, save_dir = self._set_metric_kwargs(deepcopy(self.plottable_metrics_kwargs[met.__name__]))
            tag = met.__name__ if tag is None else tag
            save_dir = self.output_dir / 'plottable_metrics' if save_dir is None else self.output_dir / save_dir
            if met_kwargs is None: fig = met(self.confusion.get_actual_vector(self.confusion_key),
                                             self.confusion.get_probability_vector(self.confusion_key), self.classes)
            else: fig = met(self.confusion.get_actual_vector(self.confusion_key),
                            self.confusion.get_probability_vector(self.confusion_key), self.classes, **met_kwargs)
            self.writer.add_figure(met.__name__, fig)
            if self.save_performance_plot: 
                if not save_dir.is_dir(): save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f'{tag}_test-epoch{self.test_epoch}.png', bbox_inches='tight')
            
    def _get_a_log(self):
        log = self.metrics.result()
        log_confusion = self.confusion.result()
        log.update(log_confusion) 
        return log
        