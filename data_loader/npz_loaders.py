import torch
import numpy as np
from torchvision import transforms
from copy import deepcopy
from base import BaseSplitDataset, DATASET_MODE, BaseSplitDatasetLoader
import data_loader.data_sampling as module_sampling

class NPZDataset(BaseSplitDataset):
    """
    npz data loading demo using BaseSplitDataLoader
    """
    def __init__(self, dataset_path:str, mode:str, trsfm=None):
        if trsfm is None:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #RGB
                # transforms.Normalize((0.5, ), (0.5, )) # Grayscale
            ])
        super().__init__(dataset_path, mode, trsfm)
        self.data, self.targets, self.classes, self.data_paths, self.paths = self._load_data_list(self.init_kwargs['dataset_path'])
    
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
        
        return item, target, '' if self.data_paths is None else self.data_paths[index]

    def _load_data_list(self, _path):
        with np.load(_path, allow_pickle=True) as file:
            try:classes = file['classes']
            except: classes = file['class_names']
            data, targets = None, None
            data_paths, paths = None, None
            for k in [k for k in file.files if self.init_kwargs['mode'] in k]:
                if any(check_item in k for check_item in ['x', 'data']): data = file[k]
                elif any(check_item in k for check_item in ['y', 'target', 'label']): targets = file[k]
                elif any(check_item in k for check_item in ['path']): data_paths = file[k]
            if data is None or targets is None:
                raise Exception(f'Only data and targets should exist. Currently found values:{file.files}')
            if data_paths is not None: paths = file['paths_per_class']
            else: print('Warning: No data path information available.')
        targets = torch.from_numpy(targets)
        return data, targets, classes, data_paths, paths

class NPZDataLoader():
    def __init__(self, dataset_path:str, batch_size:int=32, mode:list=DATASET_MODE, trsfm=None, num_workers=0, collate_fn=None, **kwargs):
        self.loaderdict = dict()
        self.size, self.classes = None, None
        
        for m in mode:
            dataset, use_kwargs = NPZDataset(dataset_path, m, trsfm), deepcopy(kwargs)
            if 'shuffle' not in kwargs.keys(): use_kwargs['shuffle'] = True if m==DATASET_MODE[0] else False
            if 'sampler' in kwargs.keys(): 
                if m == DATASET_MODE[0]:
                    if 'shuffle' in use_kwargs.keys(): use_kwargs['shuffle'] = False
                    sampling_kwargs = kwargs['sampler']['args']
                    sampling_kwargs['data_source'] = dataset
                    sampling_kwargs['classes'] = dataset.classes
                    use_kwargs['sampler'] = getattr(module_sampling, kwargs['sampler']['type'])(**sampling_kwargs)
                else: del use_kwargs['sampler']
            self.loaderdict[m] = BaseSplitDatasetLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **use_kwargs)
            print(f'Make a {m} dataloader.')
            size, classes = self._check_dataloader_shape(self.loaderdict[m])
            if self.size is None: self.size = size
            elif self.size != size: raise ValueError('The height and width sizes do not match the data loader in different modes.')
            if self.classes is None: self.classes = classes
            elif all(self.classes != classes): raise ValueError('The classes in the data loader do not match in different modes.')
        
    def _check_dataloader_shape(self, dataloader):
        X, y, path = next(iter(dataloader.dataloader))
        if X.shape[1] != 3: raise Exception(f'Shape of batch is [N, C, H, W]. Please recheck.')
        _, C, H, W = X.shape
        return (C, H, W), dataloader.dataloader.dataset.classes