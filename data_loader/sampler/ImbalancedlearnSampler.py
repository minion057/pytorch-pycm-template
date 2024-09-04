from torch.utils.data import RandomSampler
from utils import is_module_installed, check_onehot_label, integer_encoding, onehot_encoding
import imblearn
import numpy as np

class ImbalancedlearnSampler(RandomSampler):
    def __init__(self, data_source, classes,
                 sampler_type:str, sampler_name:str, sampler_kwargs:dict={}):
        if is_module_installed(f'imblearn.{sampler_type}'): module = getattr(imblearn, sampler_type)
        else: raise ValueError('The sampler_type must be "under_sampling", "over_sampling" or "combine".')
        if not hasattr(module, sampler_name): raise ValueError('The sampler_name does not exist.')
        
        # Pre Processing
        data_shape = list(data_source.data.shape)
        label_is_onehot = check_onehot_label(data_source.labels[0], classes)
        label_classes = np.unique(data_source.labels, axis=0)
        data4sampler = data_source.data.reshape(data_shape[0], -1)
        labels4sampler = integer_encoding(data_source.labels, classes)
        
        # Run
        sampler = getattr(module, sampler_name)(**sampler_kwargs)
        X, y = sampler.fit_resample(data4sampler, labels4sampler)

        # Post Processing
        data_shape[0] = len(X)
        data_source.data = X.reshape(data_shape)
        data_source.labels = onehot_encoding(y, classes) if label_is_onehot else y
        is_class_composition_equal = np.unique(data_source.labels, axis=0) == label_classes
        if is_class_composition_equal.ndim == 1: is_class_composition_equal = all(is_class_composition_equal.ndim)
        else: is_class_composition_equal = all([all(same) for same in is_class_composition_equal])
        if not is_class_composition_equal: raise ValueError('The sampled labels are not consistent with the original labels.')
        super().__init__(data_source=data_source)