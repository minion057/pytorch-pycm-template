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
    def forward_hook(self):
        def hook(module, input, output):
            raise NotImplementedError
        return hook
    
    @abstractmethod
    def forward_pre_hook(self):
        def hook(module, input):
            raise NotImplementedError
        return hook