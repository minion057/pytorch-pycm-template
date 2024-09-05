from torch.utils.data import RandomSampler
from utils import check_and_import_library, check_onehot_label, integer_encoding, onehot_encoding
import numpy as np

class ImbalancedlearnSampler(RandomSampler):
    # https://imbalanced-learn.org/stable/references/index.html
    def __init__(self, data_source, classes,
                 sampler_type:str, sampler_name:str, sampler_kwargs:dict={}):
        ''' Sampling Type
            'under_sampling': The imblearn.under_sampling provides methods to under-sample a dataset.;
            'over_sampling': The imblearn.over_sampling provides a set of method to perform over-sampling.;
            'combine': The imblearn.combine provides methods which combine over-sampling and under-sampling.;
        '''
        module = check_and_import_library(f'imblearn.{sampler_type}')
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