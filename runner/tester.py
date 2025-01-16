import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTester, MetricTracker, ConfusionTracker, BaseTracker
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

        # Additional Tracking
        self.additionalTracking_key, self.additionalTracking_columns = 'meta', ['path', 'target', 'pred']
        self.additionalTracking_columns.extend([f'prob:{c}' for c in self.classes])
        self.additionalTracking = BaseTracker(*[self.additionalTracking_key], columns=self.additionalTracking_columns)
        
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
        self.writer.set_step(self.test_epoch, self.wirter_mode)
        with torch.no_grad():
            output, use_data, use_target, use_path = [], [], [], []
            for batch_idx, load_data in enumerate(tqdm(self.data_loader)):
                if len(load_data) == 3: data, target, path = load_data
                elif len(load_data) == 2: data, target, path = load_data, None
                else: raise Exception('The length of load_data should be 2 or 3.')
                
                # 1. To move Torch to the GPU or CPU
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
                
            self.writer.add_image('input', make_grid(use_data, nrow=8, normalize=True))
            
            # 4-0. Additional tracking
            for col in self.additionalTracking_columns:
                if 'path' in col.lower() and use_path is not None: 
                    for p in use_path: self.additionalTracking.update(self.additionalTracking_key, col, str(p))
                elif 'target' in col.lower():
                    for p in confusion_content['actual']: self.additionalTracking.update(self.additionalTracking_key, col, p)
                elif 'pred' in col.lower():
                    for p in confusion_content['predict']: self.additionalTracking.update(self.additionalTracking_key, col, self.classes[p])
                elif 'prob' in col.lower():
                    class_idx = [i for i, c in enumerate(self.classes) if c in col]
                    if len(class_idx) != 1: 
                        raise ValueError(f'All class names could not be found in the name of the column ({col}) where the probability value is to be stored.')
                    for p in confusion_content['probability']: 
                        self.additionalTracking.update(self.additionalTracking_key, col, p[class_idx[0]])
        
            # 4-1. Update the Projector
            if self.projector:                    
                label_img, features = tb_projector_resize(use_data.clone(), label_img, features)
                class_labels.extend([str(self.classes[lab]) for lab in use_target])

            if self.tensorboard_pred_plot:
                self.prediction_images, self.prediction_labels = use_data[-self.preds_item_cnt:], [self.classes[lab] for lab in use_target[-self.preds_item_cnt:]]
                data_channel = self.prediction_images.shape[1]
                preds = np.squeeze(use_predict[-self.preds_item_cnt:].numpy())                    
                preds = preds if len(use_target)!=1 else np.array([preds]) # For batches with length of 1    
                use_prob = self.confusion.get_probability_vector(self.confusion_key)[-len(preds):] 
                self.prediction_preds = [self.classes[lab] for lab in preds]
                self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]  
        
        # 4-2. Update the curve plot and projector
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
        return self.criterion(output, target, self.classes, self.device)
    
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
        self._save_path_infomation()
        
    def _save_confusion_obj(self, filename='cm', message='Saving checkpoint for Confusion Matrix'):
        save_pycm_object(self.confusion.get_confusion_obj(self.confusion_key), save_dir=self.output_dir, save_name= filename)
            
    def _save_path_infomation(self, filename:str='result'):
        self.additionalTracking.save2excel(savedir=self.output_dir, savename=filename, excel_type='xlsx')