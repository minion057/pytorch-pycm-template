import torch
import numpy as np
import ctypes
import time
import datetime
from tqdm.auto import tqdm
from copy import deepcopy
from collections import OrderedDict
from torchcam import methods as Explainer
from torchcam.utils import overlay_mask
from base import BaseExplainer
from utils import write_json, reset_device, get_color_cycle, close_all_plots
from torchvision.transforms.functional import to_pil_image
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from pathlib import Path # 나중에 삭제


class TorchcamExplainer(BaseExplainer):
    """ XAI logic using TorchCAM. (https://github.com/frgfm/torch-cam) """
    def __init__(self, model, config, classes, device, 
                 explainset:dict, explain_methods:list):
        """ Logic to initialize to perform XAI using TorchCAM.

        Args:
            model: Model designed for Explainable AI (XAI) tasks
            config: Use the config.json file that was used for training the model.
            classes: Label class for the data to be used.
            device: The device used in the model. CUDA or CPU.
            explainset (dict): A dataset to perfome XAI.
                               - The key value is the index of the class, 
                               - The value is the data to use (Tensor or numpy.ndarray type), targets and paths.
            explain_methods (list): The XAI technique to perform. Get the methods supported by torchcam by name only (e.g. GradCam).
        """
        if not all(isinstance(class_idx, int) for class_idx in list(explainset.keys())): 
            print('list(explainset.keys()): ', list(explainset.keys()))
            raise ValueError('The keys in the description set must be integers: only the index of the class.')
        if 'data' not in list(explainset[0].keys()) or 'targets' not in list(explainset[0].keys()):
            raise ValueError('The explainset requires data and target information.')
        
        libc = ctypes.CDLL("libc.so.6") # for torchcam
        super().__init__(model, config, classes, device, [])
        self.config = config
        self.device = device
        self.xai_layers = None
        
        self.explain_model_name = config['arch']['type']      
        self.explainset = explainset
        self.n_samples_per_class = len(explainset[0]['index'])
        self.explain_methods = explain_methods
        
        self.gradient_based_methods = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM'] # Fast. 
        self.activation_based_methods = ['CAM', 'ScoreCAM', 'SSCAM', 'ISCAM'] # Slow.
        self.batch_enabled_methods = ['ScoreCAM', 'SSCAM', 'ISCAM'] # When using multi layer, these methods also do not support blocky layers.
        self.single_layer_only_methods = ['CAM']
        self.input_shape = tuple(self.explainset[0]['data'][0].shape) # for xai methods
        self.batch_size = 1 # for batch_enabled_methods
        self.require_single_xai_layer = any(np.isin(self.single_layer_only_methods, self.explain_methods)) # for single_layer_only_methods
        self.not_require_block_layer = any(np.isin(self.batch_enabled_methods, self.explain_methods)) # for batch_enabled_methods
        self.use_cpu_for_activation_based = False
        
        self.softmax = torch.nn.Softmax(dim=0)
        self.model.eval()
        
    def _check_xai_layers_format(self, xai_layers):
        if xai_layers is None: return
        if self.require_single_xai_layer and len(xai_layers) != 1: raise ValueError(f"Expected list of length 1, but got length {len(xai_layers)}")
        if self.not_require_block_layer:
            parent = None
            for xai_layer in xai_layers:
                # 1. The tiers that will perform XAI must have the same parent.
                layer_name_list = xai_layer.split('.')
                try:
                    layer_name_list[-1] = int(layer_name_list[-1])
                    p = '.'.join(layer_name_list[:-1])
                except:
                    p = '.'.join(layer_name_list[:-2])
                if parent is None: parent = p
                if parent != p: raise ValueError('The tiers that will perform XAI must have the same parent.')

                # 2. All tiers performing XAI must be child hierarchies.
                find_child_layer = False
                for layer_name in self.all_layers.keys():
                    layer_name_str = layer_name.replace('[', '.').replace(']', '')
                    support_name = '.'.join(layer_name_str.split('.')[:-1])
                    if xai_layer in [layer_name_str, support_name]: 
                        find_child_layer = True
                        break
                if not find_child_layer: 
                    error_message = f'{self.batch_enabled_methods} requires exact layer names in order to use multiple layers.\n'
                    error_message += 'For example, if you have layer.1.1.pw, layer.1 is not supported, but layer1.1 and layer.1.1.pw are supported.'
                    raise ValueError(error_message)

    def _explain_init(self, view_final_conv_layer_result:bool=True): # Initialize a Explainer
        if self.xai_layers is None:
            # Because the models are the same, the target layer set for torchcam will be the same for all. 
            # Therefore, we obtain the target information from the first extractor that is set up.
            explainer_args = {'model':self.model, 'target_layer':None, 'enable_hooks':False, 'input_shape':self.input_shape}
            tmp = Explainer.__dict__['GradCAM'](**explainer_args)
            self.xai_layers = tmp.target_names
            reset_device('cache', False)
        
        if view_final_conv_layer_result:
            if self.last_conv_layer_name is None: print('No convolutional hierarchy exists in the current model.')
            elif self.last_conv_layer_name != self.xai_layers[0] and self.xai_layers[0] not in self.last_conv_layer_name: 
                try:
                    tmp = deepcopy(self.xai_layers)
                    tmp.append(self.last_conv_layer_name)
                    self._check_xai_layers_format(tmp)
                    self.xai_layers.append(self.last_conv_layer_name)
                except: print('The explain_method you set does not allow you to add the last convolution by regularity.')
            else: print('The automatically set layer is the last convolutional layer, so we don\'t add it.')
            
        extractors = {}
        for idx, name in enumerate(self.explain_methods):
            # Create an extractor object for torchcam.
            explainer_args = {'model':self.model, 'target_layer':self.xai_layers, 'enable_hooks':False, 'input_shape':self.input_shape}
            if name in self.batch_enabled_methods: explainer_args['batch_size'] = self.batch_size
            if name == 'CAM': explainer_args['fc_layer'] = self._find_last_fc_layer()            
            extractors[name] = Explainer.__dict__[name](**explainer_args)
            
            
            
        _ = self._freeze_and_get_layers(self.model, self.xai_layers)
        extractors_result_template_per_data = {name:[] for name in self.explain_methods}
        extractors_result_template = {'classes':[], 'traget_layers':[], 'activation_maps':deepcopy(extractors_result_template_per_data)}
        extractors_result = OrderedDict({
            'data':[], 'labels':[], 'paths':[], 'probs':[],
            'results':deepcopy(extractors_result_template),
        })
        return extractors, extractors_result
    
    def explain(self, xai_layers:list=None, save_type:str=None, view_final_conv_layer_result:bool=True):
        """ The logic that actually performs the XAI.

        Args:
            xai_layers (list, optional): The layer to perform XAI on. Name accessible from model variables. Defaults to None.
                                         If you use the default value of None, it will be set to the last non-reduced convolutional layer.
                                         (e.g. model.layer1 -> layer1, model.layer[1] -> layer.1)
                                         Warring: If you grab the whole block, the tensor size might be different, so find the last conv layer. 
                                         If you don't want to use this part, you can comment out the if statement.
        """
        reset_device('cache', False)
        if 'deit' in self.explain_model_name.lower(): # 나중에 삭제
            xai_layers = [self.last_conv_layer_name]
        self._check_xai_layers_format(xai_layers)
        print('Run XAI using torchcam!!!')
        self.xai_layers = xai_layers # for ViT
        self.extractors, self.extractors_result = self._explain_init(view_final_conv_layer_result)
        start = time.time()
        for class_idx, class_explainset in self.explainset.items():
            for data_idx, explain_data in enumerate(class_explainset['data'], 1):
                reset_device('cache', False)
                print(f'\nCurrently, the class that performs XAI is {self.classes[class_idx]}. ({data_idx}/{len(class_explainset["data"])})')
                self.extractors_result['data'].append(explain_data)
                self.extractors_result['labels'].append(class_explainset['labels'][data_idx-1])
                if 'paths' in list(class_explainset.keys()): self.extractors_result['paths'].append(class_explainset['paths'][data_idx-1])
                
                xai_data = torch.Tensor(explain_data).to(self.device).requires_grad_(True)
                for explain_method_idx, (explain_method_name, extractor) in tqdm(enumerate(self.extractors.items())):                    
                    xai_extractor, outputs = self._explain(xai_data=xai_data, extractor=extractor, explain_method_name=explain_method_name, explain_method_idx=explain_method_idx)
                    reset_device('cache', False)
                    # Use the hooked data to compute activation map
                    try: cams = xai_extractor(class_idx, outputs)
                    except:
                        if self.device.type == 'cuda': 
                            print(f'\nWarring: CUDA out of memory. In {explain_method_name}, it is calculated in cpu.')
                            cpu_model = deepcopy(self.extractors[explain_method_name].model.to('cpu'))
                            explainer_args = {'model':cpu_model, 'target_layer':self.xai_layers, 'enable_hooks':False, 'input_shape':self.input_shape}
                            if explain_method_name in self.batch_enabled_methods: explainer_args['batch_size'] = self.batch_size
                            if explain_method_name == 'CAM': explainer_args['fc_layer'] = self._find_last_fc_layer()
                            self.extractors[explain_method_name] = Explainer.__dict__[explain_method_name](**explainer_args)
                            xai_extractor, outputs =  self._explain(xai_data=xai_data, extractor=extractor, explain_method_name=explain_method_name,explain_method_idx=explain_method_idx)
                            cams = xai_extractor(class_idx, outputs)
                        else: raise MemoryError('CUDA out of memory.')
                    for cam in cams: 
                        self.extractors_result['results']['activation_maps'][explain_method_name].append(cam.squeeze(0).cpu().squeeze(0).detach().numpy())
                    if explain_method_idx ==0:
                        for target_layer in self.xai_layers:
                            self.extractors_result['results']['classes'].append(self.classes[class_idx])
                            self.extractors_result['results']['traget_layers'].append(target_layer)
                    xai_extractor.remove_hooks()
                    xai_extractor._hooks_enabled = False
                    reset_device('cache', False)
        end = time.time()
        self._save_info(self._setting_time(start, end))
        
        reset_device('cache', False)
        if save_type is not None:
            save_type = save_type.lower()
            if save_type == 'npz': self.save_npz()
            elif save_type == 'pdf': self.generate_xai_report()
            elif save_type == 'all':
                print('Save all currently supported NPZ, PDF files.')
                self.save_npz()
                self.generate_xai_report()
            else: print('We currently only support npz and pdf files, so saving failed. If needed, please use extractors_result to save it.')
    
    def _explain(self, extractor, xai_data, explain_method_name, explain_method_idx):
        reset_device('cache', False)
        use_data = xai_data if next(extractor.model.parameters()).is_cuda else deepcopy(xai_data).cpu().requires_grad_(True)
        xai_extractor = deepcopy(extractor)
        xai_extractor._hooks_enabled = True
        
        # Prediction
        # 1. Gradient-based methods 
        if explain_method_name in self.gradient_based_methods:
            xai_extractor.model.zero_grad() # Gradient Initialization
            outputs = xai_extractor.model(use_data.unsqueeze(0)) 
        # 2. Activation-based methods
        elif explain_method_name in self.activation_based_methods:
            with torch.no_grad(): # Disable gradient calculations
                outputs = xai_extractor.model(use_data.unsqueeze(0))
        
        # Predicted probability value.
        if explain_method_idx == 0:
            probs = [self.softmax(o).tolist() for o in outputs.detach().cpu()]
            self.extractors_result['probs'].extend(np.array(probs) * 100)
        
        return xai_extractor, outputs

    def _check_result_existence(self, result=None):
        check_data = result if result is not None else self.extractors_result
        
        # Verify that each element in the list exists in a key in the dictionary.
        check_key = ['data', 'labels', 'probs', 'results']
        existence_mask = np.isin(check_key, list(check_data.keys()))
        missing_indices = np.where(~existence_mask)[0]
        if len(missing_indices) > 0:
            missing_elements = check_key[missing_indices]
            if len(missing_elements) > 0: raise KeyError(f'No {missing_elements} found.')
        
        # Check the part with the empty list.
        empty_list_elements = self._find_empty_lists(check_data)
        if len(empty_list_elements) > 0: raise ValueError(f'The {empty_list_elements} is empty.')
        return check_data    
    
    def _find_empty_lists(self, result, parent_key=''):
        empty_keys = []
        for key, value in result.items():
            if key == 'paths': continue
            full_key = f"{parent_key}[{key}]" if parent_key else key
            if isinstance(value, dict): continue #empty_keys.extend(find_empty_lists(value, full_key))
            elif isinstance(value, list) and len(value) == 0: empty_keys.append(full_key)
        return empty_keys
    
    def _save_info(self, runtime):
        print(f'runtime: {runtime}')
        content = {
            'explain_model_name': self.explain_model_name,
            'explain_methods': self.explain_methods,
            'explain_layers': self.xai_layers,
            'explainset': {
                'input_shape': self.input_shape,
                'batch_size':self.batch_size,
                'require_single_xai_layer':self.require_single_xai_layer,
                'not_require_block_layer':self.not_require_block_layer,
                'n_samples_per_class':self.n_samples_per_class,
                'index_samples_per_class': {self.classes[idx]:self.explainset[idx]['index'] for idx in self.explainset.keys()},
            },
            'runtime':runtime
        }
        write_json(content, self.output_dir / 'info.json')
    
    def _setting_time(self, start, end):        
        runtime = str(datetime.timedelta(seconds=(end - start)))
        day_time = runtime.split(' days, ')
        hour_min_sec = day_time[-1].split(":")
        if len(day_time)==2: runtime = f'{int(day_time[0])*24+int(hour_min_sec[0])}:{hour_min_sec[1]}:{hour_min_sec[-1]}'
        return runtime
    
    def save_npz(self, result=None):
        result = self._check_result_existence(result)
        save_name = self.output_dir / f'{self.n_samples_per_class}samples.npz'
        np.savez(save_name, **{str(key): value for key, value in result.items()})
        print(f'Save... {save_name}')
        
    def generate_xai_report(self, result=None, xai_layers:list=None,
                            title:str='XAI results.', items_per_page:int=5, basic_fig_size:tuple=(10,15), bar_size:float=0.5):
        result = deepcopy(self._check_result_existence(result))
        xai_layers = self.xai_layers if xai_layers is None else xai_layers
        if xai_layers is None: raise ValueError('xai_layers must not be None.')
        save_name = self.output_dir / f'{self.n_samples_per_class}samples.pdf'
        close_all_plots() # For memory management
        
        # The number of result items you should show
        num_data = len(result['data'])
        num_xai_layers = len(xai_layers)
        num_items = num_data * num_xai_layers
        items_per_row = items_per_page % num_xai_layers
        if items_per_row != 0: 
            if items_per_row == items_per_page:  items_per_page = num_xai_layers if num_xai_layers <= 5 else 5
            else: items_per_page -= items_per_row
            page_num, num_items_to_add = num_xai_layers // items_per_page, num_xai_layers % items_per_page
            if num_items_to_add != 0: page_num += 1
            print('Adjusting items_per_page.')
            print(f'We changed the count to {items_per_page} to display all results from the layers that performed XAI on {page_num} page.')        
        
        # Setting some parameters for drawing plots.
        colors = get_color_cycle()
        colors = [next(colors) for _ in self.classes]
        items_per_row = len(self.explain_methods)+2 # col = XAI results + Prediction probability results + image
        width, height = basic_fig_size
        suptitle_text_size = 10 * width
        subtitle_text_size, text_size = suptitle_text_size - width * 2.5, suptitle_text_size - width * 4
        prob_x, prob_y = np.arange(len(self.classes)), list(range(0, 101, 10))
        
        # Create a PDF
        with PdfPages(save_name) as pdf:
            now_xai_layer_idx = 0
            for idx in range(0, num_items, items_per_page):
                # Setting subplots
                data_start_idx = idx // num_xai_layers
                use_items_per_page = items_per_page if data_start_idx+(items_per_page//num_xai_layers) < num_data else (num_data-data_start_idx)*num_xai_layers
                fig = plt.figure(figsize=(items_per_row*width, use_items_per_page*height))
                gs = GridSpec(use_items_per_page, items_per_row, figure=fig, wspace=0.2, hspace=0.2)
                real_data_idx = data_start_idx-1
                
                # Drawing plots
                for row_idx in range(use_items_per_page):
                    xai_class_name = result['results']['classes'][idx+row_idx]
                    xai_target_layer = result['results']['traget_layers'][idx+row_idx]
                    if xai_target_layer == xai_layers[0]:
                        real_data_idx += 1
                        # 1. Prediction Result (Bar Chart)
                        ax = fig.add_subplot(gs[row_idx, 0])
                        ax.set_ylim(0, 100); ax.set_yticks(prob_y)
                        ax.spines['top'].set_visible(False); ax.set_yticks(prob_y, prob_y, fontdict={'fontsize':int(text_size//2)})
                        if row_idx == 0: ax.set_title("Probabilities", size=subtitle_text_size, y=1.05)
                        
                        ax.set_xticks(prob_x, self.classes, fontdict = {'fontsize' : text_size})
                        bar = ax.bar(prob_x, result['probs'][real_data_idx], width=bar_size, color=colors)
                        for v_idx, (rect, v) in enumerate(zip(bar, result['probs'][real_data_idx])):
                            h = rect.get_height()-0.1 if rect.get_height() < 100 else 98
                            h = h if h > 5 else 5
                            text_args = {'x': rect.get_x()+rect.get_width()/2.0, 'y': h, 's': '%.2f' % v,
                                            'ha': 'center', 'va': 'bottom', 'size': text_size}
                            if self.classes[v_idx] == xai_class_name: text_args['color'] = 'red'
                            ax.text(**text_args)
                        
                        # 2. Image
                        ax = fig.add_subplot(gs[row_idx, 1])
                        if row_idx == 0: ax.set_title('Input', size=subtitle_text_size, y=1.05)
                        ax.axis('off')
                        image = np.clip(result['data'][real_data_idx], 0, 255)
                        try: ax.imshow(image)            
                        except: 
                            try: ax.imshow(np.transpose(image, (1, 2, 0)))
                            except: raise Exception(f'Please check the input data shape. ({image.shape})')
                    
                    # 3. XAI Results
                    original_img = to_pil_image(result['data'][real_data_idx])
                    for col_idx, explain_method_name in enumerate(self.explain_methods, 2):
                        xai_result = result['results']['activation_maps'][explain_method_name][idx+row_idx]
                        if len(xai_result.shape) == 1:
                            ax = fig.add_subplot(gs[row_idx, 2+(len(self.explain_methods)//2)])
                            ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False); ax.set_xticklabels([]); ax.set_yticklabels([])
                            ax.text(0, 0.5, f'The result of performing XAI on layer \"{xai_target_layer}\" with the correct label, {xai_class_name} class.\nThere are no activation maps.', 
                                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=text_size, color='black')
                            break
                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        if col_idx == 2+(len(self.explain_methods)//2):
                            ax.text(0, -0.1, f'The result of performing XAI on layer \"{xai_target_layer}\" with the correct label, {xai_class_name} class.', 
                                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=text_size, color='black')
                        if row_idx == 0: 
                            explain_title = explain_method_name
                            if explain_title == 'SmoothGradCAMpp': explain_title = 'Smooth\nGradCAMpp'
                            ax.set_title(explain_title, size=subtitle_text_size, y=1.05)
                        
                        heatmap = to_pil_image(deepcopy(xai_result), mode="F")
                        ax.imshow(overlay_mask(deepcopy(original_img), heatmap, alpha=0.5)); ax.axis('off')
                d = pdf.infodict()
                d['Title'], d['Author'] = title,'https://github.com/minion057/pytorch-pycm-template'
                pdf.savefig(fig, bbox_inches='tight')
                # plt.show()
                close_all_plots()
        print(f'Save... {save_name}')
        