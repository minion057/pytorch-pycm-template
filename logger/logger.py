import logging
import logging.config
from pathlib import Path
from utils import read_json
import shutil
import os

def setup_logging(save_dir, log_config=None, default_level=logging.INFO, test_mode=False, resume_epoch=None):
    """
    Setup logging configuration
    """
    if log_config is None: log_config = f'{os.path.dirname(os.path.realpath(__file__))}/logger_config.json'
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                log_file_path = (save_dir / handler['filename']) if not test_mode else (save_dir / f"{handler['filename']}.test")

                if log_file_path.is_file():
                    if test_mode: log_file_path.unlink()
                    else:
                        ori_log_file_path = Path(f'{log_file_path}.resume.{resume_epoch}')
                        if not Path(ori_log_file_path).is_file(): shutil.copy(log_file_path, ori_log_file_path)
                        else: shutil.copy(ori_log_file_path, log_file_path)
                
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
