import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

class BaseSplitDatasetModeChecker():
    def __init__(self):
        self.training_mode_list = ['train', 'training']
        self.validation_mode_list = ['valid', 'validation']
        self.testing_mode_list = ['test', 'testing']
        
        self.mode_list = []
        self.mode_list.extend(self.training_mode_list)
        self.mode_list.extend(self.validation_mode_list)
        self.mode_list.extend(self.testing_mode_list)
    
    def checkModeInList(self, mode):
        if not any([mode.lower() == m.lower() for m in self.mode_list]): 
            raise ValueError(f'INVALID mode: "{mode}".\nPlease select the mode from the following values: [{self.mode_list}].')
    
    def isTrainingMode(self, mode):
        return any([mode.lower() == m.lower() for m in self.training_mode_list])
    def isValidationMode(self, mode):
        return any([mode.lower() == m.lower() for m in self.validation_mode_list])
    def isTestMode(self, mode):
        return any([mode.lower() == m.lower() for m in self.testing_mode_list])
    
class BaseSplitDatasetLoader():
    def __init__(self, dataset, mode:str, batch_size:int, shuffle:bool, num_workers:int, collate_fn, **kwargs):
        self.mode = mode
        self.mode_checker = BaseSplitDatasetModeChecker()
        self.mode_checker.checkModeInList(self.mode) 
        
        self.dataset = dataset
        kwargs['batch_size'] = batch_size
        kwargs['shuffle'] = shuffle
        kwargs['num_workers'] = num_workers
        kwargs['collate_fn'] = collate_fn
        use_kwargs = self._setKwargsForMode(kwargs)
        self.shuffle = use_kwargs['shuffle']
        self.dataloader = DataLoader(self.dataset, **use_kwargs)
        print(f'Make a {self.mode} dataloader.')
        
    def __len__(self):
        return len(self.dataset)

    def _setKwargsForMode(self, original_kwargs):
        use_kwargs = deepcopy(original_kwargs)
        if 'shuffle' not in original_kwargs.keys(): 
            original_kwargs['shuffle'] = True if self.mode_checker.isTrainingMode(self.mode) else False
        if 'sampler' in original_kwargs.keys(): 
            if self.mode_checker.isTrainingMode(self.mode):
                if 'shuffle' in original_kwargs.keys(): use_kwargs['shuffle'] = False
                sampling_kwargs = original_kwargs['sampler']['args']
                sampling_kwargs['data_source'] = dataset
                sampling_kwargs['classes'] = dataset.classes
                use_kwargs['sampler'] = getattr(module_sampling, original_kwargs['sampler']['type'])(**sampling_kwargs)
            else: del use_kwargs['sampler']
        return use_kwargs
    
    def _fetchDatasetInfo(self):
        _ = next(iter(self.dataloader))
        _, C, H, W = _[0].shape
        return (C, H, W), self.dataloader.dataset.classes
    
class BaseSplitDataset(Dataset):
    """
    Base class for split dataset
    Split data refers to training set, verification set, and test set.
    Alternatively, it is also applicable to consisting of a training set and a test set.
    This consists of only the desired three data sets through the parameter called `mode`.
    """
    def __init__(self, dataset_path:str, mode:str, trsfm=None):  
        self.mode_checker = BaseSplitDatasetModeChecker()
        self.mode_checker.checkModeInList(mode)
        self.init_kwargs = {
            'dataset_path': dataset_path,
            'mode': mode,
            'trsfm': trsfm
        }
        self.data, self.targets, self.classes = None, None, None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """        
        item = self.data[index]

        # You have to write a code to handle the "item".
            
        if self.init_kwargs['transform'] is not None:
            item = self.init_kwargs['transform'](item)
        target = self.targets[index]
        return item, target
        """
        raise NotImplementedError

    def _load_data_list(self, _path):
        raise NotImplementedError
