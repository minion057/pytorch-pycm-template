import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import datetime
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from copy import deepcopy
from pathlib import Path
from utils import ensure_dir, read_json, write_dict2json, convert_confusion_matrix_to_list, convert_days_to_hours
from utils import plot_confusion_matrix_1, plot_performance_N, close_all_plots

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, plottable_metric_ftns, optimizer, config, classes, device):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        
        self.classes = classes
        self.device = device
        
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.plottable_metric_ftns = plottable_metric_ftns
        self.optimizer = optimizer
        self.loss_fn_name = config['loss'] 
        self.metrics_kwargs = config['metrics'] if 'metrics' in config.config.keys() else None
        self.plottable_metrics_kwargs = config['plottable_metrics'] if 'plottable_metrics' in config.config.keys() else None
        self.confusion_key, self.confusion_tag_for_writer = 'confusion', 'ConfusionMatrix'

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.accumulation_steps = cfg_trainer['accumulation_steps'] if 'accumulation_steps' in cfg_trainer.keys() else None
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            monitor_values = self.monitor.split()
            assert monitor_values[0] in ['min', 'max']
            if len(monitor_values) == 2: 
                self.mnt_mode, self.mnt_metric, self.mnt_metric_name = monitor_values[0], monitor_values[1], None
            elif len(monitor_values) == 3:
                self.mnt_mode, self.mnt_metric, self.mnt_metric_name = monitor_values
            else:
                raise ValueError('monitor option in config file is not valid')
            
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0: self.early_stop = inf
            
        self.start_epoch = 1

        # Setting the save directory path
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = Path(config.output_dir) / 'training'
        ensure_dir(self.output_dir, True)
        self.metrics_dir = self.output_dir / 'metrics_json'
        ensure_dir(self.metrics_dir, True)
        self.output_metrics = self.metrics_dir / 'metrics.json'
        self.metrics_img_dir = self.output_dir / 'metrics_imgae'
        ensure_dir(self.metrics_img_dir)
        self.confusion_img_dir = self.output_dir / 'confusion_imgae'
        ensure_dir(self.confusion_img_dir)

        # setup visualization writer instance
        # log_dir is set to "[save_dir]/log/[name]/start_time" in advance when parsing in the config file.
        self.tensorboard = cfg_trainer['tensorboard']
        self.writer = TensorboardWriter(config.log_dir, self.logger, self.tensorboard)
        
        projector = cfg_trainer['tensorboard_projector']
        self.train_projector = projector['train']
        self.valid_projector = projector['valid']
        
        self.tensorboard_pred_plot = cfg_trainer['tensorboard_pred_plot']
        self.save_performance_plot = cfg_trainer['save_performance_plot']

        self.use_resume = False
        if config.resume is not None: 
            self.use_resume = True
            self.not_improved_cnt = 0
            self._resume_checkpoint(config.resume)
            
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number. Start with 1.s
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        start = time.time()
        not_improved_count = 0 if not self.use_resume else self.not_improved_cnt
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            result = self._train_epoch(epoch)
            epoch_end = time.time()
            runtime_per_epoch = self._setting_time(epoch_start, epoch_end)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            log['runtime'] = runtime_per_epoch
            # Save result without tensorboard
            self._save_output(log)
            # Save result with tensorboard
            self._save_tensorboard(log)

            # print logged informations to the screen
            self.logger.info('')
            self.logger.info('============ METRIC RESULT ============')
            for idx, (key, value) in enumerate(log.items(), 1):
                if '_class' in key or self.confusion_key in key: continue
                if type(value) == dict:
                    self.logger.info(f'{idx}. {str(key):15s}')
                    for i, (k, v) in enumerate(value.items(), 1):
                        self.logger.info(f'\t{idx}.{i}. {str(k):15s}: {str(v):15s}')
                else: self.logger.info(f'{idx}. {str(key):15s}: {value}')
            self.logger.info('============ METRIC RESULT ============')
            self.logger.info('')

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric (mnt_metric -> e.g., loss of validation)
                    improved = False
                    if self.mnt_mode == 'min':
                        improved = (log[self.mnt_metric] if self.mnt_metric_name is None else log[self.mnt_metric][self.mnt_metric_name]) <= self.mnt_best
                    elif self.mnt_mode == 'max':
                        improved = (log[self.mnt_metric] if self.mnt_metric_name is None else log[self.mnt_metric][self.mnt_metric_name]) >= self.mnt_best
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric] if self.mnt_metric_name is None else log[self.mnt_metric][self.mnt_metric_name]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
           
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best, not_improved_count=not_improved_count)
                self._save_other_output(epoch, log, save_best=best)
            
        end = time.time()
        self._save_runtime(self._setting_time(start, end)) # e.g., "1:42:44.046400"

    def _save_checkpoint(self, epoch, save_best=False, filename='latest', message='Saving checkpoint', not_improved_count=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        
        # Save it so that it can be loaded from a single gpu later on
        state_dict = self.model.module.state_dict() if isinstance(self.model, DP) or isinstance(self.model, DDP) else self.model.state_dict()
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'not_improved_count':not_improved_count
        }
        if filename == 'latest':
            with open(str(self.checkpoint_dir / f'{filename}.txt'), "a") as f:
                f.write(f'{filename}.pth -> epoch{epoch}\n')
        filename = str(self.checkpoint_dir / f'{filename}.pth') #'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("{}: {} ...{}".format(message, filename, '' if save_best else '\n'))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...\n")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        
        # checkpoint = torch.load(resume_path)
        # In the future, weights_only will default to True.
        # If you want to set weights_only is true, Requires the use of "torch.serialization.add_safe_globals" to set to True.
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        try: 
            self.not_improved_cnt = checkpoint['not_improved_count']
            self.logger.info("Loading Unimproved counts: {} ...".format(self.not_improved_cnt))
        except: 
            self.logger.info("Unimproved counts are not saved. So set to 0.")
            self.not_improved_cnt = 0
            
        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model.load_state_dict(checkpoint['state_dict'])
        if isinstance(self.model, DP) or isinstance(self.model, DDP): self.model.module.load_state_dict(checkpoint['state_dict'])
        else: self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}\n".format(self.start_epoch))

    def _save_other_output(self, epoch, log, save_best=False):
        pass

    def _save_tensorboard(self, log):
        # Save the value per epoch. And save the value of training andvalidation.
        if self.tensorboard:
            self.writer.set_step(log['epoch'])
            # 1. All metrics (with loss) 2. All metrics per class
            self._log_metrics_to_tensorboard(log)
            
            # 3. Confusion Matrix
            self.writer.add_figure(self.confusion_tag_for_writer, self._make_a_confusion_matrix(log[self.confusion_key]))
            if self.do_validation:
                self.writer.set_step(log['epoch'], 'valid')
                self.writer.add_figure(self.confusion_tag_for_writer, self._make_a_confusion_matrix(log[f'val_{self.confusion_key}']))
            close_all_plots()
            
    def _log_metrics_to_tensorboard(self, log:dict):
        for key, value in log.items():
            if key in ['epoch', self.confusion_key, f'val_{self.confusion_key}']: continue
            if 'val_' in key or 'time' in key: continue
            if not isinstance(value, dict): # 1. All metrics
                if f'val_{key}' in log.keys():
                    scalars = {key:value, f'val_{key}':log[f'val_{key}']}
                    self.writer.add_scalars(key, {str(k):v for k, v in scalars.items()})
                else: self.writer.add_scalar(key, value)
            else: # 2. All metrics per class
                if f'val_{key}' in log.keys():
                    scalars = deepcopy(value)
                    scalars.update({f'val_{k}':v for k, v in log[f'val_{key}'].items()})
                    self.writer.add_scalars(key, {str(k):v for k, v in scalars.items()})
                else: self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
        
    def _save_runtime(self, runtime:str):
        if not self.output_metrics.is_file(): raise ValueError('Not found output file.')
        result = read_json(self.output_metrics)
        
        if 'totaltime' in result.keys():
            prev_runtime = self._sum_timelist(result['runtime'][:self.start_epoch-1])
            runtime = self._sum_timelist([runtime, prev_runtime]) #result['totaltime']])            
        result['totaltime'] = runtime
        write_dict2json(result, self.output_metrics)

    def _setting_time(self, start, end):        
        return convert_days_to_hours(str(datetime.timedelta(seconds=(end - start))))

    def _sum_timelist(self, timelist):
        totalSecs = 0
        for tm in timelist:
            t = tm.split('.')[0]
            timeParts = [int(s) for s in t.split(':')]
            totalSecs += (timeParts[0] * 60 + timeParts[1]) * 60 + timeParts[2]
        totalSecs, sec = divmod(totalSecs, 60)
        hr, min = divmod(totalSecs, 60)
        return f'{hr:d}:{min:d}:{sec:d}'
    
    def _save_output(self, log):
        # Save the result of metrics.
        if self.output_metrics.is_file():
            result = read_json(self.output_metrics)
            result = self._slice_dict_values(result, int(log['epoch'])-1) # Adjusting the number of results per epoch.
            result = self._merge_and_append_json(result, log)
        else: result = self._convert_values_to_list(log)
        write_dict2json(result, self.output_metrics)

        # Save the result of confusion matrix image.
        self._make_a_confusion_matrix(log[self.confusion_key], save_dir=self.confusion_img_dir)
        if self.do_validation: 
            self._make_a_confusion_matrix(log[f'val_{self.confusion_key}'], save_mode='Validation', save_dir=self.confusion_img_dir)

        # Save the reuslt of metrics graphs.
        if self.save_performance_plot: plot_performance_N(result, self.metrics_img_dir/'metrics_graphs.png')
        
    def _slice_dict_values(self, content:dict, slice_size:int):
        new_content = deepcopy(content)
        for key, value in content.items():
            if key == 'totaltime': continue
            if isinstance(value, list):
                new_content[key] = value[:slice_size]
            elif isinstance(value, dict):
                new_content[key] = self._slice_dict_values(value, slice_size)
            else: raise TypeError('Restricts dictionary values to lists or dictionaries only.\n'
                                  +f'The current detected type is {type(value)}.')
        return new_content
    
    def _merge_and_append_json(self, json_data, new_data):
        use_data = deepcopy(json_data)
        for key, value in new_data.items():
            if key == 'totaltime': continue
            if key in use_data:
                if isinstance(use_data[key], list):
                    use_data[key].append(value)
                elif isinstance(use_data[key], np.ndarray):
                    use_data[key] = np.append(use_data[key], value)
                elif isinstance(use_data[key], dict):
                    if isinstance(value, dict):
                        use_data[key] = self._merge_and_append_json(use_data[key], value)
                    else:
                        raise TypeError(f"Value type mismatch for key '{key}': expected dict, got {type(value)}")
                else:
                    raise TypeError(f"Unsupported type for key '{key}': {type(use_data[key])}")
            else:
                raise KeyError(f"Key '{key}' not found in json_data")
        return use_data
    
    def _convert_values_to_list(self, content):
        new_content = deepcopy(content)
        for key, value in content.items():
            if key == 'totaltime': continue
            if isinstance(value, dict): new_content[key] = self._convert_values_to_list(value)
            else: new_content[key] = [value]
        return new_content
    
    def _sort_train_val_sequences(self, content):
        keys = list(content.keys())
        sorted_keys = []
        time_keys = []  # List to hold keys that contain 'time'
        
        for key in keys:
            if key.startswith('val_'):
                continue  # Skip 'val_' prefixed keys for now
            if 'time' in key:
                time_keys.append(key)  # Collect time-related keys
                continue
            sorted_keys.append(key)
            val_key = 'val_' + key
            if val_key in keys:
                sorted_keys.append(val_key)
        
        # Append any remaining keys that were not paired with 'val_' prefixed keys
        for key in keys:
            if key not in sorted_keys and 'time' not in key:
                sorted_keys.append(key)
        
        # Append the time-related keys at the end
        sorted_keys.extend(time_keys)
        
        # Create a new dictionary with sorted keys
        sorted_data = {key: content[key] for key in sorted_keys}
        return sorted_data
    
    def _make_a_confusion_matrix(self, confusion, class_labels:list=None,
                                 save_mode:str='Training', save_dir=None, title=None):        
        plot_kwargs = {'confusion':convert_confusion_matrix_to_list(confusion), 
                       'classes':list(confusion.keys()) if class_labels is None else class_labels}
        
        if title is not None: plot_kwargs['title'] = title
        if save_dir is None: 
            plot_kwargs['return_plot'] = True
            return plot_confusion_matrix_1(**plot_kwargs)
        else: # Save the result of confusion matrix image.
            if title is None: plot_kwargs['title'] = f'Confusion Matrix: {save_mode} Data'
            plot_kwargs['file_path'] = Path(save_dir)/f'ConfusionMatrix_{save_mode.lower().replace(" ", "_")}.png'
            plot_confusion_matrix_1(**plot_kwargs)