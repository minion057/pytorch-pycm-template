# from pycm import ROCCurve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn import metrics
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import warnings

""" Curve metric (i.g., ROC, PV) """
def check_onehot_label(item):
    item_class = np.unique(np.array(item), return_counts=True)[0]
    if type(item) not in [list, np.ndarray]: return False #print('type')
    elif len(item_class) != len(c): return False #print('class num')
    elif 0 not in item_class and 1 not in item_class: return False #print(item_class)
    else: return True

def onehot_encoding(label, classes):
    if type(classes) == np.ndarray: classes = classes.tolist() # for FutureWarning by numpy
    item = label[0]
    if item not in classes: classes = np.array([idx for idx in range(len(classes))])    
    label, classes = np.array(label), np.array(classes)
    if not check_onehot_label(item): 
        if len(classes.shape)==1: classes = classes.reshape((-1, 1))
        if len(label.shape)==1: label = label.reshape((-1, 1 if type(item) not in [list, np.ndarray] else len(item)))
        oh = OneHotEncoder()
        oh.fit(classes)
        label_onehot = oh.transform(label).toarray()
    else: label_onehot = np.array(label)
    return label_onehot

def ROC(label, prob, classes:list, specific_class_idx=None):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    label_onehot = onehot_encoding(label, classes)
    prob = np.array(prob)
    # STEP 1. micro score
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(label_onehot.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # STEP 2. macro score
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(label_onehot[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
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
    # STEP 3. class score
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
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), ncol=1) # ,mode='expand', borderaxespad=0.05
    fig.tight_layout(rect=[0, 0, 1, 1])
    return ax.figure # plot_close()

def ROC_class(label, prob, classes:list, specific_class_idx=None):
    label_onehot = onehot_encoding(label, classes)
        
    color_list = list(plt.cm.Set3.colors); del color_list[-4] # delete gray color
    color_list.extend(list(plt.cm.Pastel1.colors)); del color_list[-1]
    color_list.extend(list(plt.cm.Pastel2.colors)); del color_list[-1]
    colors = cycle(color_list)
    
    # Using pycm
    # crv = ROCCurve(actual_vector=np.array(label), probs=np.array(prob), classes=classes)
    # crv.thresholds
    # return crv #.plot(area=True, classes=specific_class_idx).figure

    # Using sklearn
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0,1],[0,1], label='y=x', color='lightgray', linestyle="--")
    for class_id, color in zip(range(len(classes)), colors):
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





'''
from sklearn.preprocessing import LabelBinarizer

def lb_encoding(label, classes:list):
    item = label[0]
    if not check_onehot_label(item): 
        if item not in classes: classes = [idx for idx in range(len(classes))]
        lb = LabelBinarizer()
        lb.fit(classes)
        label_lb = np.array(lb.transform(label))
    else: label_lb = np.array(label)
    return label_lb    
'''
