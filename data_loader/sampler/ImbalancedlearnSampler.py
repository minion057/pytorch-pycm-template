import numpy as np
from copy import deepcopy
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
        if self.paths is not None: sampled_paths = np.array(deepcopy(self.paths))
        targets2integer = self._convert_targets2integer()
        data_shape = list(self.data.shape)
        data4sampler = self.data.reshape(data_shape[0], -1)
        sampler = self._get_a_sampler()
        X, y = sampler.fit_resample(data4sampler, targets2integer)
        data_shape[0] = len(X)
        if self.paths is not None:
            ori_indices = np.array(list(range(len(data4sampler))))
            if 'under' in self.sampler_type.lower():
                use_indices = sampler.sample_indices_ # Indices of the samples selected.
                sampled_paths = sampled_paths[use_indices]
            elif 'over' in self.sampler_type.lower():
                # No information about indices is provided. 
                # But the sampled data is appended after the original data, so we fill it with the sampler name without path information
                new_paths = np.array([str(self.sampler_name)] * (len(X) - len(ori_indices)))
                sampled_paths = np.concatenate((sampled_paths, new_paths), axis=0)
            else: # combine
                # Oversampled and then undersampled -> only 2 cases (SMOTEENN, SMOTETomek)
                # So clear the path for deleted data and fill the newly created path with sampler name
                if self.sampler_name == 'SMOTEENN': use_indices = np.array(sampler.enn_.sample_indices_)
                elif self.sampler_name == 'SMOTETomek': use_indices = np.array(sampler.tomek_.sample_indices_)
                use_indices_in_real_data = use_indices[use_indices < len(ori_indices)]
                sampled_paths = sampled_paths[use_indices_in_real_data]
                new_paths = np.array([str(self.sampler_name)] * (len(X) - len(sampled_paths)))
                sampled_paths = np.concatenate((sampled_paths, new_paths), axis=0)
        self.data_source.data = X.reshape(data_shape)
        self.data_source.targets = self._revert_targets(y)
        if self.paths is not None: self.data_source.paths = sampled_paths