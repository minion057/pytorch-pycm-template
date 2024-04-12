import numpy as np
import pandas as pd
from abc import abstractmethod

class BaseHook:
    def __init__(self, *keys, cols:list, writer=None):
        self._data = pd.DataFrame(index=keys, columns=cols)
        self.writer = writer
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = None
    
    def result(self):
        return dict(self._data)
    
    @abstractmethod
    def forward_hook(self, module, input_data, output_data):
        raise NotImplementedError
        return output_data
    
    @abstractmethod
    def forward_pre_hook(self, module, input_data):
        raise NotImplementedError
        return input_data