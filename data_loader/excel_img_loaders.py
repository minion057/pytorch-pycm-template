import torch
import numpy as np
import pandas as pd
import data_loader.data_sampling as module_sampling
from torchvision import transforms
from base import BaseSplitDataset, BaseSplitDatasetLoader
from torchvision.io import read_image
from utils import onehot_encoding
from pathlib import Path

class ExcelImgDataset(BaseSplitDataset):
    """
    excel (img list) data loading demo using BaseSplitDataLoader
    """
    def __init__(self, excel_path:str, dataset_path:str, mode:str, 
                 trsfm=None, suffix:str='.png', use_onehot:bool=False):
        super().__init__(dataset_path, mode, trsfm)
        self.init_kwargs['dataset_path'] = Path(self.init_kwargs['dataset_path'])
        self.init_kwargs['excel_path'] = excel_path
        self.init_kwargs['suffix'] = suffix
        self.init_kwargs['use_onehot'] = use_onehot
        self.data_kwargs = {key:None for key in ['channel', 'height', 'width', 'mean', 'std']}
        self.data, self.targets, self.classes, self.paths, self.paths_per_class = self._load_data_list()
        
    
    def __getitem__(self, index):
        """
        This is the part that you need to customize.
        """
        item = self.data[index]
        if self.init_kwargs['trsfm'] is not None:
            item = self.init_kwargs['trsfm'](item)
        target = self.targets[index]
        filename = self.paths[index]            
        
        return item, target, filename

    def _load_data_list(self):
        meta = pd.read_excel(self.init_kwargs['excel_path'], sheet_name=None)
        
        real_classes = meta['Class Information'].loc[:, 'classes'].tolist()
        label_classes = meta['Class Information'].loc[:, 'targets'].tolist()
        paths_per_class = {k:[] for k in real_classes}
        
        use_meta = None
        for k in meta.keys():
            if self.init_kwargs['mode'].lower() in k.lower(): use_meta = meta[k]
        if use_meta is None: 
            raise ValueError(f'There is no sheet information suitable for dataset mode ({self.init_kwargs["mode"]}). Current sheet: {meta.keys().tolist()}')
        
        filename_key, class_key, subclass_key = use_meta.columns.tolist()
        
        paths = use_meta.loc[:, filename_key].tolist()
        classset = use_meta.loc[:, class_key].tolist()
        for filename, class_name in zip(paths, classset):
            paths_per_class[class_name].append(filename)
        
        targets = [label_classes[real_classes.index(class_name)] for class_name in classset]
        if self.init_kwargs['use_onehot']:
            targets = onehot_encoding(targets, real_classes)
        
        for filename, class_name, subclass_name in zip(paths, classset, use_meta.loc[:, subclass_key].tolist()):
            use_pathkey = f'{class_name}_{subclass_name}'
            if use_pathkey not in paths_per_class: paths_per_class[use_pathkey] = []
            paths_per_class[use_pathkey].append(filename)
        
        return self._loadImagesAndComputeStats(paths), np.array(targets), np.array(real_classes), np.array(paths), paths_per_class
    
    def _loadImagesAndComputeStats(self, img_paths):
        data = []
        num_channels, mean, std = None, [], []
        for filename in img_paths:
            img_path = sorted(self.init_kwargs['dataset_path'].glob(f'**/{filename}*{self.init_kwargs["suffix"]}'))
            if img_path == []: raise ValueError(f'The file "{filename}*{self.init_kwargs["suffix"]}" was not found.')
            elif len(img_path) > 1: raise ValueError(f'The file "{filename}*{self.init_kwargs["suffix"]}" is a duplicate.')
            img = read_image(str(img_path[0])) # C, H, W
            # uint8에서 float32로 변환하고 0~1 사이로 스케일링
            img = img.to(torch.float32) / 255.0
            data.append(img)
            
            if self.data_kwargs['channel'] is None:
                self.data_kwargs['channel'], self.data_kwargs['height'], self.data_kwargs['width'] = img.shape  # 이미지의 채널 수, size를 저장
                self.data_kwargs['mean'], self.data_kwargs['std'] = torch.zeros(self.data_kwargs['channel']), torch.zeros(self.data_kwargs['channel'])
                
            for i in range(self.data_kwargs['channel']):  
                self.data_kwargs['mean'][i] += img[i, :, :].mean()
                self.data_kwargs['std'][i] += img[i, :, :].std()
        
        self.data_kwargs['mean'] /= len(img_paths)
        self.data_kwargs['std'] /= len(img_paths)
        return torch.stack(data, dim=0)
        

class ExcelImgDataLoader(BaseSplitDatasetLoader):
    def __init__(self, excel_path:str, dataset_path:str, mode:str,
                 trsfm=None, suffix:str='.png', use_onehot:bool=False, 
                 batch_size:int=32, shuffle:bool=False, num_workers=0, collate_fn=None, **kwargs):  
        super().__init__(dataset=ExcelImgDataset(excel_path, dataset_path, mode, trsfm, suffix, use_onehot), mode=mode, 
                         batch_size=batch_size, shuffle=shuffle,  num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        