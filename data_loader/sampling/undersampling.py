import torch
import numpy as np

def balanced_random_undersampling(data, target):
    class_list, class_count = np.unique(target, axis=0, return_counts=True)
    min_cnt, max_cnt = min(class_count), max(class_count)
    del_class_list = class_list[list(filter(lambda x: class_count[x] == max_cnt, range(len(class_count))))]
    for del_class in del_class_list:
        del_class_item_indices = []
        for idx, item in enumerate(target):
            should_delete = torch.eq(item, torch.tensor(del_class))
            if should_delete.ndim != 0: should_delete = all(should_delete)
            if should_delete: del_class_item_indices.append(idx)
        del_class_remove_indices = np.random.choice(del_class_item_indices, max_cnt-min_cnt, replace=False)
        index_list = [x for x in range(len(target)) if x not in del_class_remove_indices]
        data, target = data[index_list], target[index_list]
    return data, target
    