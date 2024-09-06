import numpy as np
from torch.utils.data import Dataset, DataLoader

DATASET_MODE = ['train', 'valid', 'test']

class BaseSplitDatasetLoader():
    def __init__(self, dataset, batch_size:int, shuffle:bool, num_workers:int, collate_fn, **kwargs):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        
    def __len__(self):
        return len(self.dataset)

class BaseSplitDataset(Dataset):
    """
    Base class for split dataset
    Split data refers to training set, verification set, and test set.
    Alternatively, it is also applicable to consisting of a training set and a test set.
    This consists of only the desired three data sets through the parameter called `mode`.
    """
    def __init__(self, dataset_path:str, mode:str, trsfm=None):       
        # super(BaseSplitDataset, self).__init__()
        if mode not in DATASET_MODE: raise ValueError(f'INVALID mode: "{mode}".\nPlease select the mode from the following values: train, vaild, test.')
        self.init_kwargs = {
            'dataset_path': dataset_path,
            'mode': mode,
            'trsfm': trsfm
        }
        self.data, self.targets = None, None

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
