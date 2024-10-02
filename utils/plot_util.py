import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import ConfusionMatrixDisplay
from itertools import cycle

def close_all_plots():
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

def get_color_cycle():
    colors = list(plt.cm.Set3.colors)
    del colors[-4] # delete gray color
    del colors[1] # delete light yellow color
    colors.extend(list(plt.cm.Pastel1.colors))
    del colors[-1] # delete gray color
    del colors[-4] # delete light yellow color
    colors.extend(list(plt.cm.Pastel2.colors))
    del colors[-1] # delete gray color
    return cycle(colors)
    
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
    close_all_plots()
    idx = idx if len(images) >= idx else len(images)
    fig = plt.figure(figsize=(3*idx, 5))
    for i in np.arange(idx):
        ax = fig.add_subplot(1, idx, i+1, xticks=[], yticks=[])
        plot_imshow(images[i], one_channel)
        ax.set_title(f"{preds[i]}, {(probs[i] * 100.0):.1f}%\n(label: {labels[i]})",
                    color=("green" if preds[i]==labels[i] else "red"))
    if return_plot: return fig
    if show: plt.show()
    close_all_plots()

def plot_confusion_matrix_1(confusion:list, classes:list, 
                            title:str=None, file_path=None, dpi:int=300, return_plot:bool=False, show:bool=False):
    close_all_plots()
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(confusion), display_labels=np.array(classes))
    confusion_plt = disp.plot(cmap=plt.cm.binary)
    if title is not None: confusion_plt.ax_.set_title(title)
    if file_path is not None: confusion_plt.figure_.savefig(file_path, dpi=dpi, bbox_inches='tight')
    if return_plot: return confusion_plt.figure_ # close_all_plots()
    if show: plt.show()
    close_all_plots()

def plot_confusion_matrix_N(confusion_list:list, classes:list, title:str, subtitle_list:list, 
                            file_path=None, dpi:int=300, show:bool=False):
    close_all_plots()
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
    close_all_plots()
    
def plot_performance_1(logs:dict, file_path=None, figsize:tuple=None, show:bool=False):
    close_all_plots()
    color_list = list(plt.cm.Set3.colors); del color_list[-4] # delete gray color
    color_list.extend(list(plt.cm.Pastel1.colors)); del color_list[-1]
    color_list.extend(list(plt.cm.Pastel2.colors)); del color_list[-1]

    yticks, values = [], []    
    for name, score in logs.items():
        if any(item in name.lower() for item in ['epoch', 'loss', 'confusion', 'time', 'val']): continue
        if isinstance(score, dict):
            for new_name, new_score in score.items():
                yticks.append(f'{name.replace("_class", "")} {new_name}'.replace('_', ' '))
                values.append(new_score[0] if type(new_score) == list else new_score)
        else:
            yticks.append(name)
            values.append(score[0] if type(score) == list else score)
    y = np.arange(len(yticks))
    if figsize is None: figsize = (7, len(y))
    plt.figure(figsize=figsize)
    plt.title(f'Test Result\nTesting from epoch {logs["epoch"]}.', size=15)
    
    plt.subplot(1,1,1); plt.xlabel('Score', fontsize=12)
    bar = plt.barh(y, values, color=color_list, height=0.3)
    plt.xlim(0,1.05)
    for rect, v in zip(bar, values):
        score = rect.get_width()-0.1 if rect.get_width() < 1 else 0.95
        score = score if score > 0 else 0.05
        text = f'{v:.3f}'.rstrip('0').rstrip('.')
        plt.text(score, rect.get_y(), text, ha='center', va='bottom', size=12)
    plt.yticks(y, yticks, fontsize=12)
        
    plt.tight_layout()
    if file_path is not None: plt.savefig(file_path, bbox_inches='tight')
    if show: plt.show()  
    close_all_plots()

def plot_performance_N(logs:dict, file_path=None, figsize:tuple=(5,5), show:bool=False):
    close_all_plots()
    pass_items = ['epoch', 'loss', 'confusion', 'time']
    title_size, bbox_to_anchor = 15, (0,1.1,1,0.2)
    log_items = [k.lower() for k in logs.keys()]
    width, height = figsize
    row, col, dictitem = 1, 2, 0
    for name, score in logs.items():
        if any(item in name.lower() for item in pass_items): continue
        if isinstance(score, dict) and 'val' not in name.lower(): dictitem += 1
    row = row + dictitem//col + dictitem%col
    fig = plt.figure(figsize=(col*width, row*height), layout="constrained")
    gs = GridSpec(row, col, figure=fig, wspace=0.05, hspace=0.05)
    
    # 1. Basic plot: Loss information
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Loss', fontsize=title_size); ax.set_xlabel('Epochs'); ax.set_ylabel('Loss')
    ax.plot(logs['epoch'], logs['loss'], label='Loss')
    if 'val_loss'.lower() in log_items: 
        val_index = log_items.index('val_loss'.lower())
        ax.plot(logs['epoch'], logs[list(logs.keys())[val_index]], label = 'Val Loss')
    ax.legend(loc='lower left', bbox_to_anchor=bbox_to_anchor, ncol=2, mode='expand')
    if logs['epoch'][0]!=len(logs['epoch']): ax.set_xlim([logs['epoch'][0],len(logs['epoch'])])
    if len(logs['epoch']) <= 10: ax.set_xticks(logs['epoch'])
    
    # 2. Non-class versions of metrics
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('Metrics', fontsize=title_size); ax.set_xlabel('Epochs'); ax.set_ylabel('Score')
    for name, score in logs.items():
        if any(item in name.lower() for item in pass_items): continue
        elif isinstance(score, dict): continue
        ax.plot(logs['epoch'], score, label=str(name).replace('_', ' '))
    ax.legend(loc='lower left', bbox_to_anchor=bbox_to_anchor, ncol=3, mode='expand')
    if logs['epoch'][0]!=len(logs['epoch']): ax.set_xlim([logs['epoch'][0],len(logs['epoch'])])
    if len(logs['epoch']) <= 10: ax.set_xticks(logs['epoch'])
    
    # 3. Class versions of metrics
    if dictitem > 0:
        classrow, classcol = 1, 0
        for name, score in logs.items():
            if any(item in name.lower() for item in pass_items): continue
            if isinstance(score, dict) and 'val' not in name.lower(): 
                ax = fig.add_subplot(gs[classrow, classcol])
                if classcol < col-1: classcol += 1
                else: classrow, classcol = classrow+1, 0
                ax.set_title(name.replace('_class', ''), fontsize=title_size); ax.set_xlabel('Epochs'); ax.set_ylabel('Score')
                for subname, subscore in score.items():
                    ax.plot(logs['epoch'], subscore, label=str(subname).replace('_', ' '))
                if f'val_{name}'.lower() in log_items: 
                    for subname, subscore in logs[f'val_{name}'].items():
                        ax.plot(logs['epoch'], subscore, label=f'Val {subname}'.replace('_', ' '))
                ax.legend(loc='lower left', bbox_to_anchor=bbox_to_anchor, ncol=len(score) if len(score) < 4 else 3, mode='expand')
                if logs['epoch'][0]!=len(logs['epoch']): ax.set_xlim([logs['epoch'][0],len(logs['epoch'])])
                if len(logs['epoch']) <= 10: ax.set_xticks(logs['epoch'])
    
    if file_path is not None: ax.figure.savefig(file_path)
    if show: plt.show()  
    close_all_plots()
    
def show_mix_result(imgs, titles:list=['Original Data', 'Mix Data', 'Result'], cmap='viridis'):
    close_all_plots()
    if len(imgs[0]) != len(titles): raise ValueError('The number of images and titles in a row must be the same.')
    nrow, ncol = len(imgs), len(titles)
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, squeeze=False, figsize=(5*ncol,3*nrow), tight_layout=True)
    for idx, img_list in enumerate(imgs):
        for i, t in enumerate(titles):
            img = img_list[i].detach()
            img = to_pil_image(img)
            axs[idx, i].imshow(np.asarray(img), cmap=cmap)
            axs[idx, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if idx==0: axs[0, i].set_title(titles[i], size=15)
    return fig

def plot_CI(means:[list, np.ndarray], bounds:[list, np.ndarray], classes:[list, np.ndarray],
            metric_name:str, CI:int, binom_method:str, show_bound_text:bool=True,
            file_path=None, show:bool=False, return_plot:bool=True):
    if len(means) != len(bounds) != len(classes): raise ValueError('All three lists (means, bounds, and classes) must be the same length.')
    # means = np.array([float(m) if m != 'None' else 0. for m in means])
    # bounds = [tuple([float(x) if x != 'None' else 0. for x in b]) for b in bounds]
    if all(0 <= x <= 1 for x in means):
        means = np.array(means)*100
        if not all(0 <= x <= 100 for x in means): raise ValueError('The values in the list (means) are outside the range between 0 and 100.')
    if all((0 <= l <= 1 or 0 <= u <= 1) for l, u in bounds):
        bounds = np.array(bounds)*100 # [[l*100, u*100] for l, u in bounds]
        if not all((0 <= l <= 100 or 0 <= u <= 100) for l, u in bounds): raise ValueError('The values in the list (bounds) are outside the range between 0 and 100.')
    errors, real_bounds = [[], []], []
    for score, (lower, upper) in zip(means, bounds): # 메트릭 점수를 사용하기 때문에 범위를 0~100으로 고정
        lower_error, upper_error  = score - lower, upper - score
        if score-lower_error < 0 or lower_error < 0: lower_error = score
        if score+upper_error > 100: upper_error = 100 - score
        real_bounds.append((score-lower_error, score+upper_error))
        errors[0].append(lower_error); errors[1].append(upper_error)
    errors = np.array(errors)
    
    colors, palette = [], ['#D7E1EE', '#B1C9F5', '#6F8CE7', '#8B8BB7'] # ['lightsteelblue', 'cornflowerblue', 'royalblue', 'midnightblue']
    legend_kwargs = {
        'loc':'upper left', 'bbox_to_anchor':(1, 1.02), 'ncol':1,
        'handles':[mpatches.Patch(color=color) for color in palette],
        'labels':[f'{"0%":4} < {metric_name} < 25%', f'25% < {metric_name} ≤ 50%', f'50% < {metric_name} ≤ 75%', f'75% < {metric_name} ≤ 100%']
    }
    for score in means:
        if 0 <= score <= 25: colors.append(palette[0])
        elif 25 < score <= 50: colors.append(palette[1])
        elif 50 < score <= 75: colors.append(palette[2])
        else: colors.append(palette[3])
    
    close_all_plots()
    fig, ax = plt.subplots(figsize=((2 if len(means) >= 3 else 3)*len(means), 5))
    x = np.arange(len(classes))
    ax.bar(x, means, width=0.5, color=colors)
    ax.errorbar(x, means, yerr=errors, color='k', capsize=5, fmt=' ', label=f'Mean {metric_name} (±{CI}% CI)')
    ax.legend(); ax.set_ylim(0, 105)
    # plot_styles, plot_labels = plt.gca().get_legend().legendHandles, [t.get_text() for t in plt.gca().get_legend().get_texts()]
    plot_styles, plot_labels = ax.get_legend_handles_labels()
    legend_kwargs['handles'], legend_kwargs['labels'] = plot_styles + legend_kwargs['handles'], plot_labels + legend_kwargs['labels']
    ax.legend(**legend_kwargs)
    ax.set_title(f'Confidence Interval ({binom_method})', size=17, pad=20)
    ax.set_ylabel(metric_name, fontsize=15)
    if show_bound_text:
        def format_number(num):
            num = round(num, 2)
            return num if isinstance(num, float) and num % 1 != 0 else f'{int(num)}'
        ax.set_xticks([])
        rows = ['Mean', 'Upper', 'Lower']
        table_data = [[], [], []]
        for score, (down_point, up_point) in zip(means, real_bounds):
            table_data[0].append(f'{format_number(score)}%')
            table_data[1].append(f'{format_number(up_point)}%')
            table_data[2].append(f'{format_number(down_point)}%')
        table = ax.table(cellText=table_data, rowLabels=rows, colLabels=[str(c) for c in classes], loc='bottom', cellLoc='center')
        table.scale(1,2)
        table.auto_set_font_size(False)
    else: ax.xticks(x, [str(c) for c in classes], rotation=55, ha='right', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1])
    
    if file_path is not None: fig.savefig(file_path)
    if return_plot: return fig
    if show: plt.show()  
    close_all_plots()
    
def _ROC_plot_setting():
    return {
        'label_fontsize':10,
        'title_font':{'fontsize':16, 'pad':10},
        'figsize':(8,5),
        'precision':3,
        'baseline_plot_data':([0,1],[0,1]),
        'baseline_plot_args':{'color':'lightgrey', 'linestyle':'--', 'label':'y = x'},
        'macro_plot_args':{'color':'midnightblue', 'linestyle':':', 'linewidth':2},
        'micro_plot_args':{'color':'blueviolet', 'linestyle':':', 'linewidth':2},
        'roc_plot_args':{'linewidth':2},
        'legend_args':{'loc':'upper left', 'bbox_to_anchor':(1, 1.02), 'ncol':1},
        'ax_fig_tight_layout_args':{'rect':[0, 0, 1, 1]},
    }
    
def _ROC_common_plot(ax, plot_args, title:str=None, tight_layout:bool=True):
    ax.plot(*plot_args['baseline_plot_data'], **plot_args['baseline_plot_args'])
    ax.set_ylabel('Sensitivity', fontsize=plot_args['label_fontsize'])
    ax.set_xlabel(f'1 - Specificity', fontsize=plot_args['label_fontsize'])
    ax.legend(**plot_args['legend_args'])
    if title is not None: ax.set_title(title, **plot_args['title_font'])
    if tight_layout: ax.figure.tight_layout(**plot_args['ax_fig_tight_layout_args'])
    return ax

def plot_ROC(macro_fpr, macro_tpr, micro_fpr, micro_tpr, macro_area, micro_area,
             file_path=None, return_plot:bool=False, show:bool=False):
    close_all_plots()
    plot_args = _ROC_plot_setting()
    fig, ax = plt.subplots(figsize=plot_args['figsize'])
    ax.plot(macro_fpr, macro_tpr, label=f"macro-average (AUC = {macro_area:.{plot_args['precision']}f})", **plot_args['macro_plot_args'])
    ax.plot(micro_fpr, micro_tpr, label=f"micro-average (AUC = {micro_area:.{plot_args['precision']}f})", **plot_args['micro_plot_args'])
    ax = _ROC_common_plot(ax, plot_args, title='ROC Curve')
    if file_path is not None: ax.figure.savefig(file_path)
    if return_plot: return ax.figure
    if show: plt.show()  
    close_all_plots()

def plot_ROC_OvR(classes, crv, positive_class,
                 macro_fpr=None, macro_tpr=None, micro_fpr=None, micro_tpr=None, macro_area=None, micro_area=None,
                 file_path=None, return_plot:bool=False, show:bool=False):
    plot_args = _ROC_plot_setting()
    ax = crv.plot(classes=positive_class)
    if not all(value is None for value in [macro_fpr, macro_tpr, micro_fpr, micro_tpr, macro_area, micro_area]):
        ax.plot(macro_fpr, macro_tpr, label=f"macro-average (AUC = {macro_area:.{plot_args['precision']}f})", **plot_args['macro_plot_args'])
        ax.plot(micro_fpr, micro_tpr, label=f"micro-average (AUC = {micro_area:.{plot_args['precision']}f})", **plot_args['micro_plot_args'])
    ax = _ROC_common_plot(ax, plot_args, title='ROC Curve (One vs Rest)', tight_layout=False)
    ax.figure.suptitle('')
    
    # Customize legend
    plot_styles, plot_labels = ax.get_legend_handles_labels()
    plot_styles = plot_styles[-3:-1] + plot_styles[:-3] + [plot_styles[-1]]
    plot_labels = plot_labels[-3:-1] + plot_labels[:-3] + [plot_labels[-1]]
    for idx, plot_label in enumerate(plot_labels):
        if plot_label.isdigit():
            class_idx = int(plot_label)
            plot_labels[idx] = f"{classes[class_idx]} (AUC = {crv.area()[class_idx]:.{plot_args['precision']}f})"
    ax.legend(handles=plot_styles, labels=plot_labels, **plot_args['legend_args'])
    
    # return roc curve figure
    ax.figure.set_size_inches(plot_args['figsize'])
    ax.figure.tight_layout(**plot_args['ax_fig_tight_layout_args'])
    if file_path is not None: ax.figure.savefig(file_path)
    if return_plot: return ax.figure
    if show: plt.show()  
    close_all_plots()
    
def plot_ROC_OvO(classes, pos_neg_pair_indices, fpr, tpr, auc,
                 macro_pair_indices=None, macro_fpr=None, macro_tpr=None, macro_auc=None, 
                 file_path=None, return_plot:bool=False, show:bool=False):
    close_all_plots()
    # Setting up for plot
    plot_args = _ROC_plot_setting()
    width, height = plot_args['figsize']
    show_average = all(value is not None for value in [macro_pair_indices, macro_fpr, macro_tpr, macro_auc])
    if show_average: width = height+1
    row, col = 1, 2 if show_average else 1
    fig = plt.figure(figsize=(col*width, row*height), layout="constrained")
    gs = GridSpec(row, col, figure=fig, wspace=0.05, hspace=0.2)
                
    # Customize ROC curve
    common_plot_args = {'title':'ROC Curve (One vs One)', 'tight_layout':False}
    ax, colors = fig.add_subplot(gs[0, 0]), get_color_cycle()
    for idx, ((pos_class_idx, neg_class_idx), color) in enumerate(zip(pos_neg_pair_indices, colors)):
        plot_label = f'{classes[pos_class_idx]} vs {classes[neg_class_idx]} (AUC = {auc[idx]:.{plot_args["precision"]}f})'
        ax.plot(fpr[idx], tpr[idx], label=plot_label, color=color)
    ax = _ROC_common_plot(ax, plot_args, **common_plot_args)
    if show_average:
        ax.legend()
        ax = fig.add_subplot(gs[0, 1])
        for idx, ((pos_class_idx, neg_class_idx), color) in enumerate(zip(macro_pair_indices, colors)):
            plot_label = f'macro-average {classes[pos_class_idx]} and {classes[neg_class_idx]} (AUC = {macro_auc[idx]:.{plot_args["precision"]}f})'
            ax.plot(macro_fpr, macro_tpr[idx], label=plot_label, color=color)
        ax = _ROC_common_plot(ax, plot_args, **common_plot_args)
        ax.legend()
        
    # return roc curve figure
    if file_path is not None: ax.figure.savefig(file_path)
    if return_plot: return ax.figure
    if show: plt.show()  
    close_all_plots()
    