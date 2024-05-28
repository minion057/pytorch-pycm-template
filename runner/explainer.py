import torch
import numpy as np

from base import BaseExplainer

from copy import deepcopy
from collections import OrderedDict

from torchcam import methods as Explainer


from utils.utils import GPU_status_reset, createDirectory, get_model_obj, get_target_layer_list
import os
import torch
import gzip
import pickle

from torchcam.utils import overlay_mask



class TorchcamNPZExplainer(BaseExplainer):
    def __init__(self, model, config, classes, device, xai_layer_indices:list,
                 data_loader, n_samples_per_class:int, 
                 explain_methods:list, explain_pred:bool=False, prev_result:OrderedDict=None):
        super().__init__(model, config, classes, device, xai_layer_indices)
        self.config = config
        self.device = device
        
        self.explainset = self._get_a_explainset(data_loader, n_samples_per_class)
        self.explain_methods = explain_methods
        self.explain_pred = explain_pred
        
        if prev_result is not None:
            if type(prev_result) != OrderedDict:
                try: prev_result = OrderedDict(prev_result)
                except: print('The previous result is not of the OrderedDict type.')
        self.prev_result = prev_result
    
    def _get_a_explainset(self, data_loader, n_samples_per_class):
        label_classes = np.unique(data_loader.dataset.labels).tolist()
        real_classes = self.classes if label_classes[0] in self.classes else [self.classes[label_class] for label_class in label_classes]
        class_indices = {real_class:np.where(data_loader.dataset.labels == label_classes[idx])[0].tolist() for idx, real_class in enumerate(real_classes)}
        
        n_samples = {class_name:{'index':class_index[:n_samples_per_class]} for class_name, class_index in class_indices.items()}
        for class_name, class_content in n_samples.items():
            print(f'Example index list of explaination dataset: {class_name} -> {class_content["index"]}')
            data_array = np.take(data_loader.dataset.data, class_content["index"], axis=0)
            n_samples[class_name]['data'] = [to_pil_image(data) for data in data_array]
            n_samples[class_name]['label'] = np.take(data_loader.dataset.labels, class_content["index"], axis=0)
            if data_loader.dataset.data_paths is not None:
                n_samples[class_name]['path'] = np.take(data_loader.dataset.data_paths, class_content["index"], axis=0)
        return n_samples     
    
    def _explain_hook(self, explain_method_name):
        explain_model = deepcopy(self.model)
        extractor = Explainer.__dict__[explain_method_name](explain_model, target_layer=self.xai_layer, enable_hooks=False)
        return extractor
    
    def _explain_init(self): # Initialize a Explainer
        extractors = {name:self._explain_hook(name) for name in self.explain_methods}
        extractors_result_template = {target_layer:{name:[] for name in self.explain_methods} for target_layer in self.xai_layer}
        extractors_result = OrderedDict({
            'img':[], 'path':[], 'pred':[],
            # 'target':copy.deepcopy(cam_extractors), # cuda로만 불러올 수 있게 하는 원인 -> cpu 변환 필요
            'ativation':copy.deepcopy(extractors_result_template),
            'heatmap':copy.deepcopy(extractors_result_template),
            'overlay':copy.deepcopy(extractors_result_template)
        })
        if self.prev_result is not None:
            if list(self.prev_result.keys()) != list(extractors_result.keys()):
                error_message = f'There is a missing content in the explainer save file.'
                error_message+= f'\n\tOriginal keys: {list(self.prev_result.keys())}\n\tSave keys: {list(extractors_result.keys())}'
                raise ValueError(error_message)
            else: extractors_result = deepcopy(self.prev_result)
        return extractors, extractors_result
    
    def explain(self):    # 수정하세요
        extractors, extractors_result = self._explain_init()
               
        if explain_path is not None: result['path'].append(explain_path)
        explain_data = explain_data.to(use_device).requires_grad_(True)
        
        # for idx, extractor in zip(range(1, len(cam_methods) + 1), cam_extractors.values()):
        for idx, extractor in enumerate(cam_extractors.values()):
            extractor._hooks_enabled = True
            
            # Initialize a gradient of model
            m.eval()
            m.zero_grad()
            
            # Prediction
            scores = m(explain_data.unsqueeze(0))        
            
            # Option 1. Fetching in the order of classes.
            # softmax_result = torch.tensor(y_all_result['output']).softmax(dim=1) -> Training
            # sigmoid_result = torch.tensor(y_all_result['output']).sigmoid() -> 2 class
            if idx == 0: result['pred'].append(np.round(scores.sigmoid().detach().cpu().detach().numpy(), 4)*100)
            
            # Option 2. Fetching the highest probability values.
            # top_preds, top_ind= torch.topk(scores.sigmoid()[0], num_classes)
            # top_preds = np.round(top_preds.detach().cpu().detach().numpy(), 2)
            # top_ind = top_ind.tolist()
            
            # Select the class index
            class_idx = explain_classnum if not explain_pred else scores.squeeze(0).argmax().item()

            # Use the hooked data to compute activation map
            cams = extractor(class_idx, scores)
            
            for target_layer in target_layer_list:
                for cam in cams:
                    activation_map =cam.squeeze(0).cpu()
                    heatmap = to_pil_image(activation_map, mode="F")
                    overlay = overlay_mask(pil_img, heatmap, alpha=0.5)
                    result['ativation'][target_layer][extractor.__class__.__name__].append(copy.deepcopy(activation_map))
                    result['heatmap'][target_layer][extractor.__class__.__name__].append(copy.deepcopy(heatmap))
                    result['overlay'][target_layer][extractor.__class__.__name__].append(copy.deepcopy(overlay))

            # Clean data
            extractor.remove_hooks()
            extractor._hooks_enabled = False
        
        # save and compress.
        if save_dir_name is not None:
            explanations_save_dir = os.path.join(best_model_load_path, save_dir_name)
            createDirectory(explanations_save_dir)
            with gzip.open(f'{explanations_save_dir}/{model_name}-{classes[explain_classnum]}.pickle', 'wb') as fw:
                pickle.dump(result, fw)
            print(f'\t\tSave... {save_dir_name}/{model_name}-{classes[explain_classnum]}.pickle')
        if use_device == torch.device('cuda'): GPU_status_reset('cache', status_print=False)
        
        return result