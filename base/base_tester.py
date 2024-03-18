import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import abstractmethod
from logger import TensorboardWriter
from copy import deepcopy

from pathlib import Path
from utils import read_json, write_dict2json, plot_confusion_matrix_1, plot_performance_1, plot_close
import model.metric_curve_plot as module_curve_metric

import time
import datetime

class BaseTester:
    """
    Base class for all testers
    """
    def __init__(self, model, criterion, metric_ftns, curve_metric_ftns, config, classes, device):
        self.config = config
        self.logger = config.get_logger('tester', 2)
        
        self.classes = classes
        self.device = device
        
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.curve_metric_ftns = curve_metric_ftns
        self.loss_fn_name = config['loss'] 

        self.test_epoch = 1
        
        # Setting the save directory path
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = Path(config.output_dir) / 'test'
        if not self.output_dir.is_dir(): self.output_dir.mkdir(parents=True)
        self.output_metrics = self.output_dir / 'metrics-test.json'

        # setup visualization writer instance
        # log_dir is set to "[save_dir]/log/[name]/start_time" in advance when parsing in the config file.
        cfg_trainer = config['trainer']
        self.tensorboard = cfg_trainer['tensorboard']
        self.writer = TensorboardWriter(config.log_dir, self.logger, self.tensorboard)
        
        self.projector = config['tester']['tensorboard_projector']
        
        self.tensorboard_pred_plot = cfg_trainer['tensorboard_pred_plot']
        self.save_performance_plot = cfg_trainer['save_performance_plot']
        
        if config.resume is not None: self._resume_checkpoint(config.resume)
        else: self.logger.warning("Warning: Pre-trained model is not use.\n")

    @abstractmethod
    def _test(self):
        """
        Test logic
        """
        raise NotImplementedError

    def test(self):
        """
        Full test logic
        """
        start = time.time()
        result = self._test()
        end = time.time()

        # save logged informations into log dict
        log = {'epoch': self.test_epoch}
        log.update(result)
        log['runtime'] = self._setting_time(start, end)
        
        # Save result without tensorboard
        self._save_output(log)
        # Save result with tensorboard
        self._save_tensorboard(log)

        # print logged informations to the screen
        self.logger.info('')
        self.logger.info('============ METRIC RESULT ============')
        for key, value in log.items():
            if '_class' in key or 'confusion' in key: continue
            self.logger.info(f'    {str(key):15s}: {value}')
        self.logger.info('============ METRIC RESULT ============')
        self.logger.info('')     

    
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        
        # checkpoint = torch.load(resume_path)
        checkpoint = torch.load(resume_path, map_location=self.device)
            
        self.test_epoch = checkpoint['epoch']
        self.output_metrics = self.output_dir / f'metrics-test-epoch{self.test_epoch}.json'

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model.load_state_dict(checkpoint['state_dict'])
        if isinstance(self.model, DP) or isinstance(self.model, DDP): self.model.module.load_state_dict(checkpoint['state_dict'])
        else: self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info("Checkpoint loaded. Testing from epoch {}\n".format(self.test_epoch))

    def _save_output(self, log):
        # Save the result of metrics.
        result = {}
        for k, v in log.items():
            result[k] = [v]
        write_dict2json(result, self.output_metrics)

        # Save the result of confusion matrix image.
        plot_confusion_matrix_1(log['confusion'], self.classes, 'Confusion Matrix: Test Data', self.output_dir/f'confusion_matrix_test-epoch{self.test_epoch}.png')

        # Save the reuslt of metrics graphs.
        if self.save_performance_plot:
            file_name = f'metrics_graphs_test-epoch{self.test_epoch}.png'
            plot_performance_1(result, self.output_dir/file_name)

    def _save_tensorboard(self, log):
        # Save the value per epoch. And save the value of validation.
        if self.tensorboard:
            self.writer.set_step(self.test_epoch, 'test')
            for key, value in log.items():
                if key in ['epoch', 'confusion']: continue
                if 'time' in key: continue
                # 1. All metrics
                if '_class' not in key: self.writer.add_scalar(key, value)
                # 2. All metrics per class
                else: self.writer.add_scalars(key, {str(k):v for k, v in value.items()})
            
            # 3. Confusion Matrix
            self.writer.add_figure('ConfusionMatrix', plot_confusion_matrix_1(log['confusion'], self.classes, return_plot=True))
            plot_close()

    def _setting_time(self, start, end):        
        runtime = str(datetime.timedelta(seconds=(end - start)))
        day_time = runtime.split(' days, ')
        hour_min_sec = day_time[-1].split(":")
        if len(day_time)==2: runtime = f'{int(day_time[0])*24+int(hour_min_sec[0])}:{hour_min_sec[1]}:{hour_min_sec[-1]}'
        return runtime