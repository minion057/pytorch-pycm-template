import pandas as pd
import numpy as np

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if type(value) == dict:
            value = {k:(v if v != 'None' else 0) for k,v in value.items()}
            if self.writer is not None:
                self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            if self._data.counts[key] == 0: self._data.total[key] = value
            else: self._data.total[key] = {_class:(ori_score+(_score-ori_score))*n for (_class, ori_score), _score \
                                                   in zip(self._data.total[key].items(), value.values())}
            self._data.counts[key] += n
            self._data.average[key] = value
        else:
            value = value if value != 'None' else 0
            if self.writer is not None:
                self.writer.add_scalar(key, value)
            self._data.total[key] += value*n if key == 'loss' else (value-self._data.total[key])*n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key] if key == 'loss' else value
            

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)