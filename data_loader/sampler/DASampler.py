import torch
import numpy as np
from base import BaseSampler
from copy import deepcopy
from utils import check_and_import_library, integer_encoding

class DASampler(BaseSampler):   
    def _check_kwargs(self):
        if self.sampler_name.lower() in ['mixup', 'cutmix']:
            raise ValueError(f'DASampler does not support {self.sampler_name}. This can be done by utilizing a batch sampler, or by applying prob as a `None` of the DA object.')
        
        required_kwargs = {'use_balanced_sampler':True, 'sampling_strategy':'auto', 'sampling_multiplier':0.}
        if self.kwargs == {}: self.kwargs = deepcopy(required_kwargs)
        for required_param in required_kwargs.keys():
            if required_param not in self.kwargs.keys(): 
                raise ValueError(f'The required parameter ({required_param}) is not set.')
            if required_param == 'sampling_multiplier':
                if not isinstance(self.kwargs[required_param], (int, float, list)):
                    raise TypeError(f'The required parameter ({required_param}) must be int, float, list type. But got {type(self.kwargs[required_param])}.')
                if isinstance(self.kwargs[required_param], list):
                    if list(set(self.kwargs[required_param])) == [0] and not self.kwargs['use_balanced_sampler']:
                        raise ValueError(f'If use_balanced_sampler is False, the sampling_multiplier must be not equal to 0.')
                    if len(self.kwargs[required_param]) != len(self.classes):
                        error_message = f'The length of the required parameter ({required_param}) must be equal to the length of the classes.'
                        error_message += f'\nBut got {len(self.kwargs[required_param])} and {len(self.classes)}.'
                        raise ValueError(error_message)
            elif not isinstance(self.kwargs[required_param], type(required_kwargs[required_param])):
                raise TypeError(f'The required parameter ({required_param}) must be {type(required_kwargs[required_param])} type. But got {type(self.kwargs[required_param])}.')
               
    def _run(self):
        data2tensor, targets2integer = self._convert_data2tensor(), self._convert_targets2integer()
        target_indices_per_class_index = {class_index:np.where(targets2integer==class_index)[0] for class_index in range(len(self.classes))}
        class_sampling_counts = self._calculate_sampling_counts(self.kwargs['use_balanced_sampler'], targets2integer) # {class_index: num_samples_to_generate}
        sampler = self._get_a_sampler()

        sampled_data, sampled_targets = data2tensor.detach().cpu().numpy(), deepcopy(targets2integer)
        if self.paths is not None: sampled_paths = deepcopy(self.paths)
        self.da_indices = {}
        for class_index, target_indices in target_indices_per_class_index.items():
            if class_index not in class_sampling_counts: continue
            
            num_samples_to_generate = class_sampling_counts[class_index]
            sampled_indices = np.random.choice(
                target_indices, 
                size=num_samples_to_generate, 
                replace=(num_samples_to_generate > len(target_indices)) 
            )
            self.da_indices[class_index] = deepcopy(sampled_indices)
            
            numpy_data = []
            for sampled_index in sampled_indices: 
                numpy_data.append(sampler._run(data2tensor[sampled_index]).detach().cpu().numpy())
            
            sampled_data = np.concatenate((sampled_data, np.array(numpy_data)), axis=0)
            sampled_targets = np.concatenate((sampled_targets, targets2integer[sampled_indices]), axis=0)
            if self.paths is not None: 
                append_paths = [f'{self.sampler_name}_{self.paths[sampled_index]}' for sampled_index in sampled_indices]
                sampled_paths = np.concatenate((sampled_paths, append_paths), axis=0)
        self.data_source.data = self._revert_data(sampled_data)
        self.data_source.targets = self._revert_targets(sampled_targets)
        if self.paths is not None: self.data_source.paths = sampled_paths
      
    def _calculate_sampling_counts(self, using_imblearn:bool, integer_encoded_targets=None): 
        ''' 
         1. Sampling Type
            'under_sampling': The imblearn.under_sampling provides methods to under-sample a dataset.;
            'over_sampling': The imblearn.over_sampling provides a set of method to perform over-sampling.;
            'combine': The imblearn.combine provides methods which combine over-sampling and under-sampling.;
         2. Sampling Strategy
            'minority': resample only the minority class;
            'not minority': resample all classes but the minority class;
            'not majority': resample all classes but the majority class;
            'all': resample all classes;
            'auto': equivalent to 'not majority'.
        '''
        if using_imblearn: 
            if integer_encoded_targets is None: 
                raise ValueError('The integer_encoded_targets must be set via the `_convert_targets2integer` function.')
            strategy_module = check_and_import_library(f'imblearn.utils')
            if not hasattr(strategy_module, 'check_sampling_strategy'): 
                raise ValueError('This sampler requires imblearn\'s check_sampling_strategy to work.')
            counts_module =getattr(strategy_module, 'check_sampling_strategy')
            class_sampling_counts = counts_module(self.kwargs['sampling_strategy'], integer_encoded_targets, self.sampler_type)
            max_class_cnt = max(self.target_counts)
            return {class_name:int(class_cnt+(max_class_cnt*self.kwargs['sampling_multiplier'])) for class_name, class_cnt in class_sampling_counts.items()}
        else:
            class_sampling_counts = {}
            for target_class, class_cnt in zip(self.target_classes, self.target_counts):
                class_index = integer_encoding([target_class], self.classes)
                if isinstance(self.kwargs['sampling_multiplier'], (int, float)):
                    class_sampling_counts[class_index] = int(class_cnt * self.kwargs['sampling_multiplier'])
                elif isinstance(self.kwargs['sampling_multiplier'], list):
                    class_sampling_counts[class_index] = int(class_cnt * self.kwargs['sampling_multiplier'][class_index])
                else:
                    raise TypeError(f'The required parameter (`sampling_multiplier`) must be int, float, list type. But got {type(self.kwargs[required_param])}.')
            return class_sampling_counts
    
    def _get_a_sampler(self):
        module = check_and_import_library(f'data_loader.data_augmentation')
        if not hasattr(module, self.sampler_name): raise ValueError('The sampler_name does not exist.')
        sampler = getattr(module, self.sampler_name)(**self.sampler_kwargs)
        if not hasattr(sampler, '_run'): raise ValueError('The function (`_run`) that performs DA does not exist.')
        return sampler
    
    def _convert_data2tensor(self):
        converted_data = []
        for item in deepcopy(self.data):
            # RGB channel
            if len(item.shape) == 2:
                item = np.tile(np.expand_dims(item, axis=0), [3, 1, 1])
            elif len(item.shape) != 3: 
                raise ValueError('The data must have 2 or 3 dimensions.')
            else:
                if item.shape[-1] == 4: item = item[..., :3]
                item = np.moveaxis(item, -1, 0)
            converted_data.append(item)
        return torch.from_numpy(np.array(converted_data))
    
    def _revert_data(self, converted_data):
        return converted_data[:, 0, :, :]