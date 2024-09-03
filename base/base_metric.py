import pandas as pd
import numpy as np

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.index = self._data.index.values
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        if isinstance(value, dict):
            value = {k:(v if v != 'None' else 0) for k,v in value.items()}
            if self.writer is not None:
                self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            
            is_first = self._data.loc[key, 'counts'] == 0 if key in self.index else self._data.loc[key+'_'+list(value.keys())[0], 'counts'] == 0
            if is_first: 
                # self._data.total[key] = value # pandas 3.0부터 지원 X
                value_df = pd.DataFrame({'total': value.values(), 'counts':[1]*len(value), 'average': [0]*len(value)}, index=[f'{key}_{k}' for k in value.keys()])
                value_df = pd.concat([self._data, value_df])
                self._data = value_df.drop(index=(key))
                self.index = self._data.index.values
                print(self._data)
                for new_index, v in value.items(): self._data.loc[f'{key}_{new_index}', 'total'] = v
            else: 
                # self._data.loc[key, 'total'] = {_class:(ori_score+(_score-ori_score))*n for (_class, ori_score), _score in zip(self._data.loc[key, 'total'].items(), value.values())}
                for new_index, v in value.items(): 
                    prev_score = self._data.loc[f'{key}_{new_index}', 'total']
                    self._data.loc[f'{key}_{new_index}', 'total'] = (prev_score+(v-prev_score))*n
            # self._data.loc[key, 'counts'] += n
            # self._data.loc[key, 'average'] = value # pandas 3.0부터 지원 X
            for new_index, v in value.items(): 
                self._data.loc[f'{key}_{new_index}', 'counts'] += n
                self._data.loc[f'{key}_{new_index}', 'average'] = v
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
        return dict(self._data.loc[:, 'average'])