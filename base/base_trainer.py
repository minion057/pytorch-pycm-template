import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from copy import deepcopy

from pathlib import Path
from utils import read_json, write_dict2json, plot_confusion_matrix_1, plot_performance_N, close_all_plots

import time
import datetime


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
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0: self.early_stop = inf

        self.start_epoch = 1

        # Setting the save directory path
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = Path(config.output_dir) / 'training'
        if not self.output_dir.is_dir(): self.output_dir.mkdir(parents=True)
        self.output_metrics = self.output_dir / 'metrics.json'

        # setup visualization writer instance
        # log_dir is set to "[save_dir]/log/[name]/start_time" in advance when parsing in the config file.
        self.tensorboard = cfg_trainer['tensorboard']
        self.writer = TensorboardWriter(config.log_dir, self.logger, self.tensorboard)
        
        projector = cfg_trainer['tensorboard_projector']
        self.train_projector = projector['train']
        self.valid_projector = projector['valid']
        
        self.tensorboard_pred_plot = cfg_trainer['tensorboard_pred_plot']
        self.save_performance_plot = cfg_trainer['save_performance_plot']
        
        if config.resume is not None: self._resume_checkpoint(config.resume)
        
        # Sampling And DA
        self.sampling = config['data_sampling'] if 'data_sampling' in config.config.keys() else None
        if self.sampling is not None:
            self.sampling_type = str(self.sampling['type']).lower() # down or up
            self.sampling_name = str(self.sampling['name']).lower() # random, ...
        self.cfg_da = config['data_augmentation'] if 'data_augmentation' in config.config.keys() else None
        if self.cfg_da is not None:
            self.DA = str(self.cfg_da['type'])#.lower()
            self.DAargs, self.hookargs = self.cfg_da['args'], self.cfg_da['hook_args']
            self.pre_hook = self.cfg_da['hook_args']['pre']
    
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
        not_improved_count = 0
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
                if '_class' in key or 'confusion' in key: continue
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
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
           
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            
        end = time.time()
        self._save_runtime(self._setting_time(start, end)) # e.g., "1:42:44.046400"

    def _save_checkpoint(self, epoch, save_best=False, filename='latest', message='Saving checkpoint'):
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
            'config': self.config
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
        checkpoint = torch.load(resume_path, map_location=self.device)
            
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

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

    def _save_output(self, log):
        # log = {'epoch':1, metrics:1, val_metrics:2, confusion:3, val_confusion:4}
        
        # Save the result of metrics.
        if self.output_metrics.is_file():
            result = read_json(self.output_metrics)
            for k in result.keys():
                if k == 'totaltime': continue
                if len(result[k]) != int(log['epoch']): result[k]=result[k][:int(log['epoch'])-1]
                result[k].append(log[k])
        else:
            result = {}
            for k, v in log.items():
                result[k] = [v]
        write_dict2json(result, self.output_metrics)

        # Save the result of confusion matrix image.
        plot_confusion_matrix_1(log['confusion'], self.classes, 'Confusion Matrix: Training Data', self.output_dir/'confusion_matrix_training.png')
        if 'val_confusion' in list(log.keys()): plot_confusion_matrix_1(log['val_confusion'], self.classes, 'Confusion Matrix: Validation Data', self.output_dir/'confusion_matrix_validation.png')

        # Save the reuslt of metrics graphs.
        if self.save_performance_plot: plot_performance_N(result, self.output_dir/'metrics_graphs.png')

    def _save_tensorboard(self, log):
        # Save the value per epoch. And save the value of training andvalidation.
        if self.tensorboard:
            self.writer.set_step(log['epoch']-1)
            for key, value in log.items():
                if key in ['epoch', 'confusion', 'val_confusion']: continue
                if 'val_' in key or 'time' in key: continue
                # 1. All metrics
                if '_class' not in key:
                    if f'val_{key}' in log.keys():
                        content = {key:value, f'val_{key}':log[f'val_{key}']}
                        self.writer.add_scalars(key, {str(k):v for k, v in content.items()})
                    else: self.writer.add_scalar(key, value)
                # 2. All metrics per class
                else:
                    if f'val_{key}' in log.keys():
                        content = deepcopy(value)
                        content.update({f'val_{k}':v for k, v in log[f'val_{key}'].items()})
                        self.writer.add_scalars(key, {str(k):v for k, v in content.items()})
                    else: self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            
            # 3. Confusion Matrix
            self.writer.add_figure('ConfusionMatrix', plot_confusion_matrix_1(log['confusion'], self.classes, return_plot=True))
            if 'val_confusion' in log.keys():
                self.writer.set_step(log['epoch']-1, 'valid')
                self.writer.add_figure('ConfusionMatrix', plot_confusion_matrix_1(log['val_confusion'], self.classes, return_plot=True))
            close_all_plots()
        
    def _save_runtime(self, runtime:str):
        if not self.output_metrics.is_file(): raise ValueError('Not found output file.')
        result = read_json(self.output_metrics)
        
        if 'totaltime' in result.keys():
            prev_runtime = self._sum_timelist(result['runtime'][:self.start_epoch-1])
            runtime = self._sum_timelist([runtime, prev_runtime]) #result['totaltime']])            
        result['totaltime'] = runtime
        write_dict2json(result, self.output_metrics)

    def _setting_time(self, start, end):        
        runtime = str(datetime.timedelta(seconds=(end - start)))
        day_time = runtime.split(' days, ')
        hour_min_sec = day_time[-1].split(":")
        if len(day_time)==2: runtime = f'{int(day_time[0])*24+int(hour_min_sec[0])}:{hour_min_sec[1]}:{hour_min_sec[-1]}'
        return runtime

    def _sum_timelist(self, timelist):
        totalSecs = 0
        for tm in timelist:
            t = tm.split('.')[0]
            timeParts = [int(s) for s in t.split(':')]
            totalSecs += (timeParts[0] * 60 + timeParts[1]) * 60 + timeParts[2]
        totalSecs, sec = divmod(totalSecs, 60)
        hr, min = divmod(totalSecs, 60)
        return f'{hr:d}:{min:d}:{sec:d}'