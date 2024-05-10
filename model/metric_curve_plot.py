from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn import metrics

from pycm import ROCCurve

from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from copy import deepcopy
import warnings

""" 
Curve metric (i.g., ROC, PV)
"""

""" 1. Drawing a typical ROC curve """
def check_onehot_label(item, classes):
    item_class = np.unique(np.array(item), return_counts=True)[0]
    if type(item) == int: return False
    elif len(item_class) != len(classes): return False #print('class num')
    elif 0 not in item_class and 1 not in item_class: return False #print(item_class)
    else: return True

def onehot_encoding(label, classes):
    if type(classes) == np.ndarray: classes = classes.tolist() # for FutureWarning by numpy
    item = label[0]
    if not check_onehot_label(item, classes): 
        if item not in classes: classes = np.array([idx for idx in range(len(classes))])   
        label, classes = np.array(label), np.array(classes)
        if len(classes.shape)==1: classes = classes.reshape((-1, 1))
        if len(label.shape)==1: label = label.reshape((-1, 1 if type(item) not in [list, np.ndarray] else len(item)))
        oh = OneHotEncoder()
        oh.fit(classes)
        label_onehot = oh.transform(label).toarray()
    else: label_onehot = np.array(label)
    return label_onehot

def colors():
    colors = list(plt.cm.Set3.colors); del colors[-4] # delete gray color
    colors.extend(list(plt.cm.Pastel1.colors)); del colors[-1]
    colors.extend(list(plt.cm.Pastel2.colors)); del colors[-1]
    return cycle(colors)

def ROC(label, prob, classes:list, specific_class_idx=None):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    thresholds, best_threshold_idx = dict(), dict()
    
    label_onehot = onehot_encoding(label, classes)
    prob = np.array(prob)
    
    # STEP 1. class score
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(label_onehot[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        tpr_fpr = tpr[i] - fpr[i]
        best_threshold_idx[i] = np.argmax(tpr_fpr)
        
    # STEP 2. macro score
    fpr_grid = np.linspace(0.0, 1.0, 1000)    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)    
    for i in range(len(classes)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= len(classes)
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    best_threshold_idx["macro"] = np.argmax(tpr["macro"] - fpr["macro"])
    
    # STEP 3. micro score
    fpr["micro"], tpr["micro"], thresholds["micro"] = metrics.roc_curve(label_onehot.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    best_threshold_idx["micro"] = np.argmax(tpr["micro"] - fpr["micro"])
    
    # STEP 4. plot
    fig, ax = ROC_class(label, prob, classes)    
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=2,
    )    
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=2,
    ) 
    
    # To find the best point, uncomment the relevant section and use it.
    # marker_list = cycle([k for k in Line2D.markers.keys() if k not in ['None', 'none', ' ', '']])
    # for (key, best_idx), marker, color in zip(best_threshold_idx.items(), marker_list, colors()):
    #     best_sen = tpr[key][best_idx]
    #     best_str = f'sensitivity = {best_sen:.3f}'
    #     if key not in ['macro', 'micro']:
    #         best_spec = 1-fpr[key][best_idx]
    #         best_str = f'Best Threshold of {classes[key]} | {best_str}, specificity = {best_spec:.3f} Threshold={thresholds[key][best_idx]:.3f}'
    #     else:
    #         best_spec = fpr[key][best_idx]
    #         best_str = f'Best Threshold of {key} | {best_str}, specificity = {best_spec:.3f}'
    #     ax.scatter(best_spec, best_sen, marker=marker, s=100, color=color, label=best_str)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), ncol=1) # ,mode='expand', borderaxespad=0.05
    fig.tight_layout(rect=[0, 0, 1, 1])
    return ax.figure # plot_close()


def ROC_class(label, prob, classes:list, specific_class_idx=None):
    label_onehot = onehot_encoding(label, classes)       
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0,1],[0,1], label='y=x', color='lightgray', linestyle="--")
    for class_id, color in zip(range(len(classes)), colors()):
        RocCurveDisplay.from_predictions(
            label_onehot[:, class_id],
            prob[:, class_id],
            name=f"ROC curve of {classes[class_id]}",
            linewidth=1.5,
            color=color,
            ax=ax,
        )    
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    return fig, ax     # plot_close()
""" 1. Drawing a typical ROC curve """

""" 2. Drawing a Fixed specificity ROC curve """
def FixedNegativeROC(label, prob, classes:list, negative_class_idx:int=0):
    # Setting up for use with `pycm` 
    if type(label[0]) in [list, np.ndarray]: 
        if type(label[0]) == np.ndarray: label = label.tolist()
        label = [a.index(1.) for a in label]
    elif type(label[0]) == 'str': label = [self.classes.index(a) for a in label]
    if len(prob[0]) != len(classes): raise ValueError('Need probability values for each class.')
    actual_prob = [p[a] for p, a in zip(prob, label)]
    crv = ROCCurve(actual_vector=np.array(label), probs=np.array(prob), classes=np.unique(label).tolist())
    
    # Setting up for plot
    label_fontsize = 11
    title_fontsize, title_pad = 14, 10
    pos_classes = {idx:name for idx, name in enumerate(classes) if idx != negative_class_idx}
    
    # Show
    roc_plot = crv.plot(classes=pos_classes.keys())
    roc_plot.set_ylabel('Sensitivity', fontsize=label_fontsize)
    roc_plot.set_xlabel(f'1 - Specificity\n(Negative Class is {classes[negative_class_idx]})', fontsize=label_fontsize)
    roc_plot.figure.suptitle('')
    roc_plot.set_title('ROC Curve', fontsize=title_fontsize, pad=title_pad)
    new_legend = []
    for l in roc_plot.legend().texts:
        class_idx = int(l.get_text())
        new_legend.append(f'{classes[class_idx]} (Area = {crv.area()[class_idx]:.3f})')
    new_legend.append('y = x')
    roc_plot.legend(labels=new_legend, loc='upper left', bbox_to_anchor=(1, 1.02), ncol=1)
    roc_plot.figure.tight_layout(rect=[0, 0, 1, 1])
    return roc_plot.figure # plot_close()
""" 2. Drawing a Fixed specificity ROC curve """
