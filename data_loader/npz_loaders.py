import torch
import numpy as np
from torchvision import transforms
from base import BaseSplitDataset, BaseSplitDatasetLoader
from copy import deepcopy
import data_loader.data_sampling as module_sampling

class NPZDataset(BaseSplitDataset):
    """
    npz data loading demo using BaseSplitDataLoader
    """
    def __init__(self, dataset_path:str, mode:str, trsfm=None, del_classes:list=None):
        if trsfm is None:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #RGB
                # transforms.Normalize((0.5, ), (0.5, )) # Grayscale
            ])
        super().__init__(dataset_path, mode, trsfm)
        self.data, self.targets, self.classes, self.paths, self.paths_per_class = self._load_data_list(self.init_kwargs['dataset_path'])
        if del_classes is not None: self._del_info_using_classes(del_classes)
    
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
            if paths is not None: paths_per_class = file['paths_per_class'].item()
            else: print('Warning: No data path information available.')
        targets = torch.from_numpy(targets)
        return data, targets, classes, paths, paths_per_class
    
    def _del_info_using_classes(self, del_classes):
        paths_per_class_copy = deepcopy(self.paths_per_class)
        print(f'Original classes information: {self.paths_per_class.keys()}')
        
        keys_to_delete, paths_to_remove = [], []
        for key, paths_in_key in paths_per_class_copy.items():
            if any(del_class in key for del_class in del_classes):
                keys_to_delete.append(key)
                paths_to_remove.extend(paths_in_key)
                print(f'Found del key: {key} (items: {len(paths_in_key)})')
        print(f'Keys to be deleted: {keys_to_delete}')
        print(f'Number of items to delete: {len(paths_to_remove)}')
        
        paths_to_remove_set = set(paths_to_remove)
        current_paths = self.paths.tolist()
        indices_to_delete = [
            i for i, path in enumerate(current_paths)
            if path in paths_to_remove_set
        ]
        indices_to_delete.sort(reverse=True)
        print(f'Indices to be deleted in data: {len(indices_to_delete)}')

        mask = np.ones(len(self.data), dtype=bool)
        mask[indices_to_delete] = False
        print(f'Number of items to delete: {len(indices_to_delete)}')
        del_x = self.data[mask]
        print(f'Results of deletion from X:     {len(self.data)} > {len(del_x)}')
        del_y = self.targets[mask]
        print(f'Results of deletion from y:     {len(self.targets)} > {len(del_y)}')
        del_paths = self.paths[mask]
        print(f'Results of deletion from paths: {len(self.paths)} > {len(del_paths)}')
        
        found_lingering_paths = False
        for key, paths_in_key in self.paths_per_class.items():
            if key in keys_to_delete: 
                del paths_per_class_copy[key] # Delete the key from the dictionary
                continue
            # Check which of these paths are also in the paths_to_remove_set
            still_matching_paths = [path for path in paths_in_key if path in paths_to_remove_set]
            
            if still_matching_paths:
                found_lingering_paths = True
                print(f'[WARNING] Key "{key}" in "paths_per_class" still contains paths that were targeted for deletion:') 
                print(f'  Number of matching paths: {len(still_matching_paths)}') 
                print(f'  Matching paths: {still_matching_paths}') 
                print(f'  This information will be removed from "paths_per_class" only, as the actual data/targets/paths should already be deleted.')
                
                # Remove the matching paths from the current paths_in_key list
                # Update paths_per_class_copy directly
                paths_per_class_copy[key] = [path for path in paths_in_key if path not in paths_to_remove_set]
                print(f'  Updated key "{key}" in "paths_per_class": {len(paths_in_key)} -> {len(paths_per_class_copy[key])}')
        if not found_lingering_paths:
            print("No lingering paths found in 'paths_per_class' related to the deleted classes. (Expected behavior)") 
            
        self.data, self.targets, self.paths, self.paths_per_class = del_x, del_y, del_paths, paths_per_class_copy
        print(f'Change classes information: {self.paths_per_class.keys()}')

class NPZDataLoader(BaseSplitDatasetLoader):
    def __init__(self, dataset_path:str, mode:str, trsfm=None, del_classes:list=None,
                 batch_size:int=32, shuffle:bool=False, num_workers=0, collate_fn=None, **kwargs):       
        super().__init__(dataset=NPZDataset(dataset_path, mode, trsfm, del_classes), mode=mode, 
                         batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        
    def _setKwargsForMode(self, original_kwargs):
        use_kwargs = deepcopy(original_kwargs)
        use_kwargs['shuffle'] = True if self.mode_checker.isTrainingMode(self.mode) else False
        if 'sampler' in original_kwargs.keys(): 
            if self.mode_checker.isTrainingMode(self.mode):
                if 'shuffle' in original_kwargs.keys(): use_kwargs['shuffle'] = False
                sampling_kwargs = original_kwargs['sampler']['args']
                sampling_kwargs['data_source'] = self.dataset
                sampling_kwargs['classes'] = self.dataset.classes
                use_kwargs['sampler'] = getattr(module_sampling, original_kwargs['sampler']['type'])(**sampling_kwargs)
            else: del use_kwargs['sampler']
        return use_kwargs
    
    def _cal_mean_std(self):
        if 'train' not in self.mode.lower():
            raise Exception('Mean and standard deviation can only be calculated for train mode.')
        
        mean_sum = 0.0
        std_sum = 0.0
        total_pixels = 0 # Total number of pixels for each channel

        for items in self.dataloader:
            images = items[0]
            # Reshapes the image tensor to (batch_size, num_channels, num_pixels).
            images = images.view(images.size(0), images.size(1), -1) 
            
            # Accumulates the mean and standard deviation for each channel in the current batch.
            # Sum of means across pixels for each channel
            mean_sum += images.mean(2).sum(0) # (C,)
            # Sum of standard deviations across pixels for each channel
            std_sum += images.std(2).sum(0) # (C,)
            # Total number of pixels across all channels in the batch
            total_pixels += images.size(0) * images.size(2) # (B * H*W)

        # Calculates the channel-wise mean and standard deviation for all images.
        calculated_mean = mean_sum / len(self.dataset.data)
        # Divide by the number of images to get the mean
        calculated_std = std_sum / len(self.dataset.data)
        # Divide by the number of images to get the standard deviation
        
        return calculated_mean, calculated_std
    