import pandas as pd
import numpy as np
from copy import deepcopy

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._init_data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._index_separator = '_'
        self.reset()

    def reset(self):
        self._data = deepcopy(self._init_data)
        self.index = self._data.index.values
        self._original_to_combined_index = {}
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        if isinstance(value, dict):
            value = {k:(v if v != 'None' else 0) for k,v in value.items()}
            if self.writer is not None: self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            
            is_first = self._data.loc[key, 'counts'] == 0 if key in self.index else self._data.loc[self._original_to_combined_index[key][0], 'counts'] == 0
            if is_first: 
                self._original_to_combined_index[key] = [f'{key}{self._index_separator}{k}' for k in value.keys()]
                value_df = pd.DataFrame({'total': value.values(), 'counts':[1]*len(value), 'average': [0]*len(value)}, index=self._original_to_combined_index[key])
                value_df = pd.concat([self._data, value_df])
                self._data = value_df.drop(index=(key)) if key in value_df.index.values else value_df
                self.index = self._data.index.values
                for new_index, v in zip(self._original_to_combined_index[key], value.values()): self._data.loc[new_index, 'total'] = v
            else: 
                for new_index, v in zip(self._original_to_combined_index[key], value.values()): 
                    prev_score = self._data.loc[new_index, 'total']
                    self._data.loc[new_index, 'total'] = (prev_score+(v-prev_score))*n
            for new_index in self._original_to_combined_index[key]: 
                self._data.loc[new_index, 'counts'] += n
                self._data.loc[new_index, 'average'] = self._data.loc[new_index, 'total']
        else:
            value = value if value != 'None' else 0
            if self.writer is not None:
                self.writer.add_scalar(key, value)
            self._data.loc[key, 'total'] += value*n if key == 'loss' else (value-self._data.loc[key, 'total'])*n
            self._data.loc[key, 'counts'] += n
            self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts'] if key == 'loss' else value
        
    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        result_dict = dict(self._data.loc[:, 'average'])
        for original_key, modifiy_keys in self._original_to_combined_index.items():
            original_result = {key.split(self._index_separator)[-1]:result_dict[key] for key in modifiy_keys}
            for key in modifiy_keys: del result_dict[key]
            result_dict[original_key] = original_result
        return result_dict