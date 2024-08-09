import numpy as np
from sklearn import metrics
from pycm import ROCCurve
from pycm import ConfusionMatrix as pycmCM
from pycm.pycm_util import thresholds_calc, threshold_func
from utils import onehot_encoding, integer_encoding
from utils import plot_CI, plot_ROC, plot_ROC_OvR, plot_ROC_OvO
from itertools import combinations, permutations
from copy import deepcopy
from base import base_class_metric

""" 
Metrics that need to be plotted, such as CI and ROC curves.
"""

""" CI (Confidence interval) """
def CI_wilson_class(labels, probs, classes=None, 
                    use_metric:str='ACC', alpha:float=0.05, binom_method:str='wilson', one_sided:bool=False, 
                    return_result:bool=False, plot_metric_name:str=None):
    labels, probs = integer_encoding(labels, classes), np.array(probs)
    labels, preds = [classes[idx] for idx in labels], [classes[idx] for idx in np.argmax(probs, axis=1)]
    confusion_obj = pycmCM(actual_vector=np.array(labels), predict_vector=np.array(preds))
    if use_metric is None: raise ValueError('CI requires use_metric.')
    if alpha >= 1 or alpha <= 0: raise ValueError('CI requires alpha between 0 and 1.')
    kwargs = {'param':use_metric, 'alpha':alpha, 'binom_method':binom_method, 'one_sided':one_sided}
    bounds = base_class_metric('CI', confusion_obj, classes, **kwargs)
    means = base_class_metric(use_metric, confusion_obj, classes)
    result = {class_name:{f'mean':mean, 'lower_ci':ci_bounds[0], 'upper_ci':ci_bounds[1]} for (se, ci_bounds), (class_name, mean) in zip(bounds.values(), means.items())}
    fig = plot_CI(list(means.values()), [ci_bound for se, ci_bound in bounds.values()], classes=classes, 
                  metric_name=use_metric if plot_metric_name is None else plot_metric_name, 
                  CI=int((1-alpha)*100), binom_method=binom_method, return_plot=True)
    if return_result: return result, fig
    else: return fig
""" CI (Confidence interval) """


""" 
Curve metric (i.g., ROC, PV)
"""    
def ROC(labels, probs, classes:list, crv=None):
    """ 1. Drawing a ROC curve using average (macro/micro) """
    return_average_value = False if crv is None else True
    if crv is None:
        labels, probs = integer_encoding(labels, classes), np.array(probs) # only integer label
        label_classes = np.unique(labels).tolist()
        crv = ROCCurve(actual_vector=np.array(labels), probs=np.array(probs), classes=label_classes)
    macro_fpr = np.linspace(0.0, 1.0, len(crv.data[0]['FPR'])) 
    macro_tpr = np.zeros_like(macro_fpr) 
    for class_idx in crv.data.keys(): 
        fpr, tpr = crv.data[class_idx]['FPR'], crv.data[class_idx]['TPR']
        fpr.reverse(); tpr.reverse()
        macro_tpr += np.interp(macro_fpr, fpr, tpr)
    macro_tpr /= len(crv.data.keys())
    labels2onehot = onehot_encoding(labels, classes)
    micro_fpr, micro_tpr, _thresholds = metrics.roc_curve(labels2onehot.ravel(), probs.ravel())
    macro_area, micro_area = metrics.auc(macro_fpr, macro_tpr), metrics.auc(micro_fpr, micro_tpr)
    if return_average_value: return macro_fpr, macro_tpr, micro_fpr, micro_tpr, macro_area, micro_area
    return plot_ROC(macro_fpr, macro_tpr, micro_fpr, micro_tpr, macro_area, micro_area, return_plot=True)

def ROC_OvR(labels, probs, classes:list, positive_class_indices:[int, list, np.ndarray]=None, 
            show_average:bool=False, return_result:bool=False):
    """ 2. Drawing a ROC curve using One vs Rest """
    labels, probs = integer_encoding(labels, classes), np.array(probs) # only integer label
    label_classes = np.unique(labels).tolist()
    crv = ROCCurve(actual_vector=np.array(labels), probs=np.array(probs), classes=label_classes)
    # Setting up for plot
    if not isinstance(positive_class_indices, (int, list, np.ndarray)) and positive_class_indices is not None:
        raise TypeError("positive_class_indices must be an int, list, or np.ndarray")
    if isinstance(positive_class_indices, (int)): positive_class_indices = [positive_class_indices]
    positive_class = label_classes if positive_class_indices is None else positive_class_indices
    
    # Customize ROC curve
    plot_kwargs = {'classes':classes, 'crv':crv, 'positive_class':positive_class, 'return_plot':True}
    if show_average:
        macro_fpr, macro_tpr, micro_fpr, micro_tpr, macro_area, micro_area = ROC(labels, probs, classes, crv)
        plot_kwargs.update({'macro_fpr':macro_fpr, 'macro_tpr':macro_tpr, 'macro_area':macro_area,
                            'micro_fpr':micro_fpr, 'micro_tpr':micro_tpr, 'micro_area':micro_area})
    fig = plot_ROC_OvR(**plot_kwargs)
    if return_result: 
        roc_dict = {'pos_neg_idx':[], 'fpr':[], 'tpr':[], 'auc':[], 'threshold':[], 'actual':[], 'prob':[]}
        for pos_class_idx in positive_class:
            roc_dict['pos_neg_idx'].append([pos_class_idx, None])
            roc_dict['fpr'].append(crv.data[pos_class_idx]['FPR'][:-1]) # 0 -> 1
            roc_dict['tpr'].append(crv.data[pos_class_idx]['TPR'][:-1]) # 0 -> 1
            roc_dict['auc'].append(list(crv.area().values())[pos_class_idx])
            roc_dict['threshold'].append(crv.thresholds)
            roc_dict['actual'].append(crv.actual_vector)
            roc_dict['prob'].append(crv.probs)
        return {k:np.array(v) for k, v in roc_dict.items()}, fig
    else: return fig

def ROC_OvO(labels, probs, classes:list, 
            positive_class_indices:[int, list, np.ndarray]=None, negative_class_indices:[int, list, np.ndarray]=None,
            show_average:bool=False, return_result:bool=False):
    def _roc_data(actual_vector, predict_vector, class_list, pos_class_name, thresholds=None):
        fpr, tpr, thresholds = [], [], thresholds_calc(predict_vector) if thresholds is None else thresholds
        for t in thresholds:
            def lambda_fun(x): return threshold_func(x, pos_class_name, class_list, t)
            cm = pycmCM(actual_vector=actual_vector, predict_vector=predict_vector, threshold=lambda_fun)
            fpr.append(cm.FPR[pos_class_name]); tpr.append(cm.TPR[pos_class_name])
        return np.array(fpr), np.array(tpr), np.array(thresholds)
    
    """ 3. Drawing a ROC curve using One vs One """
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-using-the-ovo-macro-average
    labels, probs = integer_encoding(labels, classes), np.array(probs) # only integer label
    label_classes = np.unique(labels).tolist()
    
    # print(f'init class_indices: {positive_class_indices} & {negative_class_indices}')
    # Calculate ROC curve for each class combination
    if positive_class_indices is None and negative_class_indices is None: 
        raise ValueError("positive_class_indices and negative_class_indices cannot be None at the same time.")
    if positive_class_indices is not None:
        if not isinstance(positive_class_indices, (int, list, np.ndarray)): 
            raise TypeError("positive_class_indices must be an int, list, or np.ndarray")
        if isinstance(positive_class_indices, (int)): positive_class_indices = [positive_class_indices]
        if len(np.where(np.array(positive_class_indices)>=len(classes))[0]) != 0: 
            raise ValueError('The positive_class_indices cannot be outside of its length of a class.')
    else: positive_class_indices = deepcopy(label_classes)
    if negative_class_indices is not None:
        if not isinstance(negative_class_indices, (int, list, np.ndarray)): 
            raise TypeError("negative_class_indices must be an int, list, or np.ndarray")
        if isinstance(negative_class_indices, (int)): negative_class_indices = [negative_class_indices]
        if len(np.where(np.array(negative_class_indices)>=len(classes))[0]) != 0: 
            raise ValueError('The negative_class_indices cannot be outside of its length of a class.')
        # Can count all classes, so comment out the two lines below.
        # pos_neg_same_index = [index for index, element in enumerate(negative_class_indices) if element in positive_class_indices]
        # if pos_neg_same_index != []: raise ValueError(f'The class to be set negative cannot be the same as the class to be set positive.: {np.array(classes)[pos_neg_same_index]}')    
    else: negative_class_indices = deepcopy(label_classes)
    # print(f'change class_indices: {list(positive_class_indices)} & {list(negative_class_indices)}')
    
    cal_pair_list, need_pair_list = np.array(list(combinations(label_classes, 2))), np.array(list(permutations(label_classes, 2)))
    # print(f'init pair_list: {list(cal_pair_list)} & {list(need_pair_list)}')
    cal_pair_list = [i for i in cal_pair_list if i[0] in positive_class_indices or i[1] in positive_class_indices]
    cal_pair_list = [i for i in cal_pair_list if i[0] in negative_class_indices or i[1] in negative_class_indices]
    need_pair_list = need_pair_list[np.isin(need_pair_list[:, 0], positive_class_indices)]
    need_pair_list = need_pair_list[np.isin(need_pair_list[:, 1], negative_class_indices)]
    if list(cal_pair_list) == [] or list(need_pair_list) == []: raise ValueError('No ROC curve can be drawn.')
    # print(f'change pair_list: {list(cal_pair_list)} & {list(need_pair_list)}')
    
    roc_dict = {'fpr':[], 'tpr':[], 'auc':[], 'thresholds':[], 'actual':[], 'prob':[]}
    positive_roc_dict, negative_roc_dict = deepcopy(roc_dict), deepcopy(roc_dict)
    mean_roc_dict = {'fpr':np.linspace(0.0, 1.0, 1000), 'tpr':[], 'auc':[]}
    for (pos_class_idx, neg_class_idx) in cal_pair_list:
        pos_mask, neg_mask = labels == pos_class_idx, labels == neg_class_idx
        all_mask = np.logical_or(pos_mask, neg_mask)
        all_idx = np.flatnonzero(all_mask)
        
        pos_labels, pos_probs = pos_mask[all_mask], probs[all_idx, pos_class_idx]
        neg_labels, neg_probs = neg_mask[all_mask], probs[all_idx, neg_class_idx]
        
        # sklearn has many times fewer thresholds than pycm, so we use pycm to get more information. 
        # If you want to use sklearn, replace the commented sk variables with pycm variables.
        fpr_pos, tpr_pos, threshold_pos = _roc_data(pos_labels, pos_probs, [False, True], True)
        fpr_neg, tpr_neg, threshold_neg = _roc_data(neg_labels, neg_probs, [False, True], True)
        # fpr_pos_sk, tpr_pos_sk, threshold_pos_sk = metrics.roc_curve(pos_labels, pos_probs) 
        # fpr_neg_sk, tpr_neg_sk, threshold_neg_sk = metrics.roc_curve(neg_labels, neg_probs)
        auc_pos, auc_neg = metrics.auc(fpr_pos, tpr_pos), metrics.auc(fpr_neg, tpr_neg)
        
        positive_roc_dict['fpr'].append(fpr_pos); negative_roc_dict['fpr'].append(fpr_neg)
        positive_roc_dict['tpr'].append(tpr_pos); negative_roc_dict['tpr'].append(tpr_neg)
        positive_roc_dict['auc'].append(auc_pos); negative_roc_dict['auc'].append(auc_neg)
        positive_roc_dict['thresholds'].append(threshold_pos); negative_roc_dict['thresholds'].append(threshold_neg)
        positive_roc_dict['actual'].append(pos_labels); negative_roc_dict['actual'].append(neg_labels)
        positive_roc_dict['prob'].append(pos_probs); negative_roc_dict['prob'].append(neg_probs)
        
        if show_average:
            mean_roc_dict['tpr'].append(np.zeros_like(mean_roc_dict['fpr']))
            mean_roc_dict['tpr'][-1] += np.interp(mean_roc_dict['fpr'], np.flip(fpr_pos), np.flip(tpr_pos))
            mean_roc_dict['tpr'][-1] += np.interp(mean_roc_dict['fpr'], np.flip(fpr_neg), np.flip(tpr_neg))
            # mean_roc_dict['tpr'][-1] += np.interp(mean_roc_dict['fpr'], fpr_pos_sk, tpr_pos_sk)
            # mean_roc_dict['tpr'][-1] += np.interp(mean_roc_dict['fpr'], fpr_neg_sk, tpr_neg_sk)
            mean_roc_dict['tpr'][-1] /= 2
            mean_roc_dict['auc'].append(metrics.auc(mean_roc_dict['fpr'], mean_roc_dict['tpr'][-1]))
    
    roc_dict['pos_neg_idx'] = []
    for (pos_class_idx, neg_class_idx) in need_pair_list:
        same_index = next((i for i, pair in enumerate(cal_pair_list) if np.array_equal(pair, [pos_class_idx, neg_class_idx])), -1)
        reverse_index = next((i for i, pair in enumerate(cal_pair_list) if np.array_equal(pair, [neg_class_idx, pos_class_idx])), -1)
        if same_index == -1 and reverse_index == -1: raise ValueError(f'Did not calculate ROC.: {classes[pos_class_idx]} vs {classes[neg_class_idx]}')
        use_index = same_index if same_index != -1 else reverse_index
        use_roc = positive_roc_dict if same_index != -1 else negative_roc_dict
        roc_dict['pos_neg_idx'].append([pos_class_idx, neg_class_idx])
        roc_dict['fpr'].append(use_roc['fpr'][use_index])
        roc_dict['tpr'].append(use_roc['tpr'][use_index])
        roc_dict['auc'].append(use_roc['auc'][use_index])
        roc_dict['thresholds'].append(use_roc['thresholds'][use_index])
        roc_dict['actual'].append(use_roc['actual'][use_index])
        roc_dict['prob'].append(use_roc['prob'][use_index])
    plot_kwargs = {'classes': classes, 'pos_neg_pair_indices':roc_dict['pos_neg_idx'],
                   'fpr': roc_dict['fpr'], 'tpr': roc_dict['tpr'], 'auc': roc_dict['auc'], 'return_plot': True}
    if show_average:
        plot_kwargs.update({'macro_pair_indices':cal_pair_list, 'macro_auc': mean_roc_dict['auc'],
                            'macro_fpr': mean_roc_dict['fpr'], 'macro_tpr': mean_roc_dict['tpr']})
    fig = plot_ROC_OvO(**plot_kwargs)
    # return roc curve figure
    if return_result: return roc_dict, fig
    else: return fig