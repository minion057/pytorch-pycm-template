import numpy as np
from abc import abstractmethod
from torch.utils.data import RandomSampler
from utils import check_onehot_encoding_1, integer_encoding, onehot_encoding

class BaseSampler(RandomSampler):
    def __init__(self, data_source, classes,
                 sampler_type:str, sampler_name:str, sampler_kwargs:dict={}, **kwargs):
        self.data_source, self.classes = data_source, list(classes)
        self.sampler_type, self.sampler_name= sampler_type, sampler_name
        self.sampler_kwargs, self.kwargs = sampler_kwargs, kwargs
        
        self._set_data_and_targets()
        self._set_paths()
        self._check_kwargs()
        self._run()
        self._set_data_and_targets()
        self._set_paths()
        
        super().__init__(data_source=data_source)
    
    def _set_data_and_targets(self):
        '''
        You need to set up a `self.data` and `self.targets'.
        '''
        try: self.data, self.targets = self.data_source.data, self.data_source.targets
        except: raise ValueError('data_source must have `data` and `targets` attributes.')
        
    def _set_paths(self):
        try: self.paths = self.data_source.paths
        except: self.paths = None
    
    def _check_kwargs(self):
        pass
    
    @abstractmethod
    def _run(self):
        raise NotImplementedError
    
    def _convert_targets2integer(self):
        if not hasattr(self, 'data') or not hasattr(self, 'targets'): \
            raise RuntimeError('The data and targets must be set via the `_set_data_and_targets` function.')
        self.target_is_onehot = check_onehot_encoding_1(self.targets[0], self.classes)
        self.target_classes, self.target_counts = np.unique(self.targets, axis=0, return_counts=True)
        return integer_encoding(self.targets, self.classes)
        
    def _revert_targets(self, converted_targets):
        if not hasattr(self, 'target_is_onehot') or not hasattr(self, 'target_classes'): 
            raise RuntimeError('The `_convert_targets2integer` function was not executed.')
        reverted_targets = onehot_encoding(converted_targets, self.classes) if self.target_is_onehot else converted_targets
        is_class_composition_equal = np.unique(reverted_targets, axis=0) == self.target_classes
        if is_class_composition_equal.ndim == 1: is_class_composition_equal = all(is_class_composition_equal)
        else: is_class_composition_equal = all([all(same) for same in is_class_composition_equal])
        if not is_class_composition_equal: raise ValueError('The reverted targets are not consistent with the original targets.')
        return reverted_targets
    