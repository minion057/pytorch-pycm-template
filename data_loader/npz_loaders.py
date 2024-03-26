import torch
import numpy as np
from torchvision import transforms

from base import BaseSplitDataset, DATASET_MODE, BaseSplitDatasetLoader

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
        self.data, self.labels, self.classes = self._load_data_list(self.init_kwargs['dataset_path'])
    
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
        label = self.labels[index]
        return item, label

    def _load_data_list(self, _path):
        with np.load(_path, allow_pickle=True) as file:
            classes = file['class_names']
            path_per_class = file['path_per_class']
            data, labels, path = None, None, None
            for k in [k for k in file.files if self.init_kwargs['mode'] in k]:
                if 'x' in k or 'data' in k: data = file[k]
                elif 'y' in k or 'label' in k: labels = file[k]
                elif 'path' in k: path = file[k] 
            if data is None or labels is None or path is None:
                raise Exception(f'Only data, labels, and file paths should exist. Currently found values:{keys}')
        labels = torch.from_numpy(labels)
        return data, labels, classes#, path, path_per_class

class NPZDataLoader():
    def __init__(self, dataset_path:str, batch_size:int=32, mode:list=DATASET_MODE, trsfm=None, num_workers=0, collate_fn=None, **kwargs):
        self.loaderdict = dict()
        self.size, self.classes = None, None
        for m in mode:
            self.loaderdict[m] = BaseSplitDatasetLoader(NPZDataset(dataset_path, m, trsfm), batch_size, \
                                                        True if m==DATASET_MODE[0] else False, num_workers, collate_fn, **kwargs)
            print(f'Make a {m} dataloader.')
            size, classes = self._check_dataloader_shape(self.loaderdict[m])
            if self.size is None: self.size = size
            elif self.size != size: raise ValueError('The height and width sizes do not match the data loader in different modes.')
            if self.classes is None: self.classes = classes
            elif all(self.classes != classes): raise ValueError('The classes in the data loader do not match in different modes.')
        
    def _check_dataloader_shape(self, dataloader):
        X, y = next(iter(dataloader.dataloader))
        if X.shape[1] != 3: raise Exception(f'Shape of batch is [N, C, H, W]. Please recheck.')
        _, C, H, W = X.shape
        return (C, H, W), dataloader.dataloader.dataset.classes