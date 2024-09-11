from base import BaseSampler
from utils import check_and_import_library

class ImbalancedlearnSampler(BaseSampler):
    # https://imbalanced-learn.org/stable/references/index.html    
    def _get_a_sampler(self):
        ''' Sampling Type
            'under_sampling': The imblearn.under_sampling provides methods to under-sample a dataset.;
            'over_sampling': The imblearn.over_sampling provides a set of method to perform over-sampling.;
            'combine': The imblearn.combine provides methods which combine over-sampling and under-sampling.;
        '''
        module = check_and_import_library(f'imblearn.{self.sampler_type}')
        if not hasattr(module, self.sampler_name): raise ValueError('The sampler_name does not exist.')
        return getattr(module, self.sampler_name)(**self.sampler_kwargs)
    
    def _run(self):
        targets2integer = self._convert_targets2integer()
        data_shape = list(self.data.shape)
        data4sampler = self.data.reshape(data_shape[0], -1)
        X, y = self._get_a_sampler().fit_resample(data4sampler, targets2integer)
        data_shape[0] = len(X)
        self.data_source.data = X.reshape(data_shape)
        self.data_source.targets = self._revert_targets(y)