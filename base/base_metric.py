import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from pycm import ConfusionMatrix as pycmCM
import matplotlib.pyplot as plt

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if type(value) == dict:
            value = {k:(v if v != 'None' else 0) for k,v in value.items()}
            if self.writer is not None:
                self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            if self._data.counts[key] == 0: self._data.total[key] = value
            else: self._data.total[key] = {_class:(ori_score+(_score-ori_score))*n for (_class, ori_score), _score \
                                                   in zip(self._data.total[key].items(), value.values())}
            self._data.counts[key] += n
            self._data.average[key] = value
        else:
            value = value if value != 'None' else 0
            if self.writer is not None:
                self.writer.add_scalar(key, value)
            self._data.total[key] += value*n if key == 'loss' else (value-self._data.total[key])*n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key] if key == 'loss' else value
            

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class ConfusionTracker:
    def __init__(self, *keys, writer=None, classes=None):
        self.writer = writer
        self.classes = classes
        self._data = pd.DataFrame(index=keys, columns=['actual', 'predict', 'probability', 'confusion'])
        self.reset()

    def reset(self):
        for idx in self._data.index.values:
            self._data['actual'][idx], self._data['predict'][idx], self._data['probability'][idx] = [], [], []
            self._data['confusion'][idx] = [np.zeros((len(self.classes),), dtype=int).tolist() for _ in range(len(self.classes))]

    def update(self, key, value:dict, set_title:str=None, img_save_dir_path:str=None, img_update:bool=False):
        if 'actual' not in list(value.keys()) or 'predict' not in list(value.keys()):
            raise ValueError(f'Correct answer (actual), predicted value (predict) and option value (probability) are required to update ConfusionTracker.\nNow Value {list(value.keys())}.')
        self._data.actual[key].extend(value['actual'])
        self._data.predict[key].extend(value['predict'])
        if 'probability' in value.keys(): self._data.probability[key].extend(value['probability'])
        confusion_obj = pycmCM(actual_vector=self._data.actual[key], predict_vector=self._data.predict[key])
        self._data.confusion[key] = confusion_obj.to_array().tolist()

        if img_update or set_title is not None or img_save_dir_path is not None:
            # Perform only when all classes of data are present
            if len(self.classes) != len(np.unique(np.array(self._data.confusion[key]), return_counts=True)[0]): return            
            confusion_plt = self.createConfusionMatrix(self._data.confusion[key])
            confusion_plt.ax_.set_title(set_title if set_title is not None else f'Confusion matrix - {key}')
        
        if self.writer is not None and img_update:
            print('ConfusionTracker: ', img_update)
            self.writer.add_figure('ConfusionMatrix', confusion_plt.figure_)
        if img_save_dir_path is not None:
            confusion_plt.figure_.savefig(os.path.join(img_save_dir_path, f'ConfusionMatrix{key}.png'),dpi=300)

    def get_actual_vector(self, key):
        return list(self._data.actual[key])
    def get_prediction_vector(self, key):
        return list(self._data.predict[key])
    def get_probability_vector(self, key):
        return list(self._data.probability[key])
    def get_confusion_matrix(self, key):
        return dict(self._data.confusion[key])
    def get_confusion_obj(self, key):
        return pycmCM(actual_vector=self.get_actual_vector(key), predict_vector=self.get_prediction_vector(key))
    def result(self):
        return dict(self._data.confusion)

    def createConfusionMatrix(self, value):
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(value), display_labels=np.array(self.classes))
        confusion_plt = disp.plot(cmap=plt.cm.binary)
        return confusion_plt