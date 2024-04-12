import numpy as np

def random_downsampling(data, target):
    class_list, class_count = np.unique(target, return_counts=True)
    min_cnt, max_cnt = min(class_count), max(class_count)
    down_class_list = class_list[list(filter(lambda x: class_count[x] == max_cnt, range(len(class_count))))]
    for down_class in down_class_list:
        down_class_item_index = np.where(np.array(target) == down_class)[0]
        down_class_remove_index = np.random.choice(down_class_item_index, max_cnt-min_cnt, replace=False)
        index_list = [x for x in range(len(target)) if x not in down_class_remove_index]
        data, target = data[index_list], target[index_list]
    return data, target
    