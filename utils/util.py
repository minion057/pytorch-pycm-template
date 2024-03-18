import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def write_dict2json(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as handle:
        json.dump(content, handle, ensure_ascii=False, indent='\t')
        
def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.\n")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def reset_device(mode):
    if mode not in ['memory', 'cache']:
        raise ValueError(f'INVALID mode: "{mode}"')
    if mode == 'memory':
        print('You can delete unnecessary variables.')
    else:
        print(f'\nThe garbage collector is currently running.')
        print(f'Before cache size : {round(torch.cuda.memory_reserved()/1024**3,1)}GB.')
        torch.cuda.empty_cache()
        print(f'After cache size : {round(torch.cuda.memory_reserved()/1024**3,1)}GB.\n')

def tb_projector_resize(data, label_img, features):           
    # probs 추가하고 싶으면 metadata_header, zip list 이용해서 수정
    _, c, h, w = data.shape
    resize_h = h
    while True:
        if resize_h < 30: break
        resize_h = resize_h//2  
    data = torch.nn.functional.interpolate(data, (resize_h), mode='bilinear', align_corners=False, antialias=True) 
    _, c, h, w = data.shape                   
    label_img = torch.cat((label_img, data), 0) if label_img is not None else data
    features = torch.cat((features, data.clone().view(-1, c*h*w)), 0) if features is not None else data.clone().view(-1, c*h*w) #28*28 -> 90MB
    return label_img, features
        
def plot_close():    
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

def plot_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def plot_classes_preds(images, labels, preds, probs, idx:int=5, one_channel:bool=False, return_plot:bool=False, show:bool=False):
    idx = idx if len(images) <= idx else len(images)
    fig = plt.figure(figsize=(3*idx, 5))
    for i in np.arange(idx):
        ax = fig.add_subplot(1, idx, i+1, xticks=[], yticks=[])
        plot_imshow(images[i], one_channel)
        ax.set_title(f"{preds[i]}, {(probs[i] * 100.0):.1f}%\n(label: {labels[i]})",
                    color=("green" if preds[i]==labels[i] else "red"))
    if return_plot: return fig
    if show: plt.show()
    plot_close()

def plot_ROC_curve(self, confusion, save:bool=False):
    # fpr, tpr
    self.ROC['FPR'].append(fpr)
    self.ROC['TPR'].append

def plot_confusion_matrix_1(confusion:list, classes:list, title:str=None, file_path=None, dpi:int=300, return_plot:bool=False, show:bool=False):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confusion), display_labels=np.array(classes))
    confusion_plt = disp.plot(cmap=plt.cm.binary)
    if title is not None: confusion_plt.ax_.set_title(title)
    if file_path is not None: confusion_plt.figure_.savefig(file_path, dpi=dpi, bbox_inches='tight')
    if return_plot: return confusion_plt.figure_ # plot_close()
    if show: plt.show()
    plot_close()

def plot_confusion_matrix_N(confusion_list:list, classes:list, title:str, subtitle_list:list, file_path=None, dpi:int=300, show:bool=False):
    if len(confusion_list) <= 1: raise ValueError('If you have a confusion matrix, you can to use \'write_confusion_matrix_1\'.')
    figsize = (len(confusion_list)*3,5)
    f, axes = plt.subplots(1, len(confusion_list), figsize=figsize, sharey=True, sharex=True, tight_layout=True)
    for i, confusion in enumerate(confusion_list):
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confusion), display_labels=np.array(classes))
        disp.plot(ax=axes[i], xticks_rotation=45, cmap=plt.cm.binary)
        disp.ax_.set_title(subtitle_list[i])
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        if i!=0: disp.ax_.yaxis.set_ticks_position('none')

    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    # common color bar
    cbar_ax = f.add_axes([1.02, 0.25, 0.02, 0.5])
    f.colorbar(disp.im_, cax=cbar_ax)
    # common title
    f.suptitle(title, fontsize=15, y=0.9)
    # common axis labels
    f.supxlabel('Predicted label', y=0.1)
    f.supylabel('True label', x=0)
    plt.tight_layout(pad=0.5, h_pad=3)
    if file_path is not None: f.savefig(file_path, dpi=dpi, bbox_inches='tight')
    if show: plt.show()  
    plot_close()
    
def plot_performance_1(logs:dict, file_path=None, figsize:tuple=None, show:bool=False):
    color_list = list(plt.cm.Set3.colors); del color_list[-4] # delete gray color
    color_list.extend(list(plt.cm.Pastel1.colors)); del color_list[-1]
    color_list.extend(list(plt.cm.Pastel2.colors)); del color_list[-1]

    xticks, values = [], []    
    for name, score in logs.items():
        if name in ['epoch', 'loss', 'val_loss', 'confusion', 'val_confusion']: continue
        if '_class' in name or 'time' in name: continue
        xticks.append(name)
        values.append(score[0])
    x = np.arange(len(xticks))
    if figsize is None: figsize = (len(x)*1.5, 5)
    plt.figure(figsize=figsize)
    plt.suptitle('Test Result', size=15)
    
    plt.subplot(1,1,1)
    plt.title(f'Testing from epoch {logs["epoch"]}.'); plt.xlabel('Metrics'); plt.ylabel('Score')
    
    bar = plt.bar(x, values, color=color_list, width=0.45)
    plt.ylim(0,1.05)
    for rect, v in zip(bar, values):
        height = rect.get_height()-0.1 if rect.get_height() < 1 else 0.95
        height = height if height > 0 else 0.1
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % v, ha='center', va='bottom', size=12)
    plt.xticks(x, xticks)
        
    plt.tight_layout()
    if file_path is not None: plt.savefig(file_path, bbox_inches='tight')
    if show: plt.show()  
    plot_close()

def plot_performance_N(logs:dict, file_path=None, figsize:tuple=(15,5), show:bool=False):
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.title('Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.plot(logs['epoch'], logs['loss'], label='Loss')
    if 'val_loss' in list(logs.keys()): plt.plot(logs['epoch'], logs['val_loss'], label = 'Val_Loss')
    plt.legend(loc='lower left', bbox_to_anchor=(0,1.15,1,0.2), ncol=2, mode='expand')
    if logs['epoch'][0]!=len(logs['epoch']): plt.xlim([logs['epoch'][0],len(logs['epoch'])])
    if len(logs['epoch']) <= 10: plt.xticks(logs['epoch'])
    
    plt.subplot(1,2,2)
    plt.title('Metrics'); plt.xlabel('Epochs'); plt.ylabel('Score')
    for name, score in logs.items():
        if name in ['epoch', 'loss', 'val_loss', 'confusion', 'val_confusion']: continue
        if '_class' in name or 'time' in name: continue
        plt.plot(logs['epoch'], score, label=str(name))
    plt.legend(loc='lower left', bbox_to_anchor=(0,1.15,1,0.2), ncol=4, mode='expand')
    if logs['epoch'][0]!=len(logs['epoch']): plt.xlim([logs['epoch'][0],len(logs['epoch'])])
    if len(logs['epoch']) <= 10: plt.xticks(logs['epoch'])
        
    plt.tight_layout()
    if file_path is not None: plt.savefig(file_path, bbox_inches='tight')
    if show: plt.show()  
    plot_close()