from base import BaseSampler
import imblearn
from utils import is_module_installed, integer_encoding

class ImbalancedlearnSampler(BaseSampler):
    def __iter__(self, data_source , labels,
                 sampler_type:str, sampler_name:str, sampler_kwargs:dict={}):
        if sampler_type not in ['under_sampling', 'over_sampling', 'combine']:
            raise ValueError('The sampler_type must be "under_sampling", "over_sampling" or "combine".')
        if is_module_installed(f'imblearn.{sampler_type}'): module = getattr(imblearn, sampler_type)
        if not hasattr(modulea, sampler_name): raise ValueError('The sampler_name does not exist.')
        
        sampler = getattr(module, sampler_name)(**sampler_kwargs)
        dataset4sampler = dataset.reshape(len(labels), -1)
        labels4sampler = integer_encoding(labels, classes)
        X, y = sampler.fit_resample(dataset4sampler, labels4sampler)