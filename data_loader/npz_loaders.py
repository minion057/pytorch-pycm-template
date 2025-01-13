import torch
import numpy as np
from torchvision import transforms
from base import BaseSplitDataset, BaseSplitDatasetLoader
import data_loader.data_sampling as module_sampling

class NPZDataset(BaseSplitDataset):
    """
    npz data loading demo using BaseSplitDataLoader
    """
    def __init__(self, dataset_path:str, mode:str, trsfm=None):
        if trsfm is None:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #RGB
                # transforms.Normalize((0.5, ), (0.5, )) # Grayscale
            ])
        super().__init__(dataset_path, mode, trsfm)
        self.data, self.targets, self.classes, self.paths, self.paths_per_class = self._load_data_list(self.init_kwargs['dataset_path'])
    
    def __getitem__(self, index):
        """
        This is the part that you need to customize.
        """
        item = self.data[index]

        # RGB channel
        if len(item.shape) == 2:
            item = np.tile(np.expand_dims(item, axis=-1), [1, 1, 3])
        elif len(item.shape) == 3 and item.shape[-1] == 4:
            item = item[..., :3]
            
        if self.init_kwargs['trsfm'] is not None:
            item = self.init_kwargs['trsfm'](item)
        target = self.targets[index]
        
        return item, target, f'Data {index}' if self.paths is None else self.paths[index]

    def _load_data_list(self, _path):
        with np.load(_path, allow_pickle=True) as file:
            try:classes = file['classes']
            except: classes = file['class_names']
            data, targets = None, None
            paths, paths_per_class = None, None
            for k in [k for k in file.files if self.init_kwargs['mode'] in k]:
                if any(check_item in k for check_item in ['x', 'data']): data = file[k]
                elif any(check_item in k for check_item in ['y', 'target', 'label']): targets = file[k]
                elif any(check_item in k for check_item in ['path']): paths = file[k]
            if data is None or targets is None:
                raise Exception(f'Only data and targets should exist. Currently found values:{file.files}')
            if paths is not None: paths_per_class = file['paths_per_class']
            else: print('Warning: No data path information available.')
        targets = torch.from_numpy(targets)
        return data, targets, classes, paths, paths_per_class

class NPZDataLoader(BaseSplitDatasetLoader):
    def __init__(self, dataset_path:str, mode:str, trsfm=None,
                 batch_size:int=32, shuffle:bool=False, num_workers=0, collate_fn=None, **kwargs):       
        super().__init__(dataset=NPZDataset(dataset_path, mode, trsfm), mode=mode, 
                         batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, **kwargs)