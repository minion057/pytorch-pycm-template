import torch
import numpy as np
import pandas as pd
from torch import nn
from base import BaseTester
from utils import check_onehot_encoding_1, ensure_dir
from tqdm.auto import tqdm
from pathlib import Path

class TesterExcel(BaseTester):
    """
    Trainer class
    """
    def __init__(self, model, config, classes, device, data_loader, is_test:bool=True):
        self.config = config
        self.logger = config.get_logger('tester', 2)
        
        self.data_loader = data_loader
        self.classes = classes
        self.device = device
        self.model = model

        self.test_epoch = 1
        self.test_dir_name = 'test' if is_test else 'valid'

        # Setting the save directory path
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = Path(config.output_dir) / self.test_dir_name / f'epoch{self.test_epoch}'
        
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
            self.output_dir = Path(config.output_dir) / self.test_dir_name / f'epoch{self.test_epoch}'
        else: self.logger.warning("Warning: Pre-trained model is not use.\n")
        
        ensure_dir(self.output_dir)
        self.filename_key, self.pred_key, self.prob_key = 'filename', 'predict', 'probability'
        self.tracker = {
            self.filename_key: [],
            self.pred_key: [],
            self.prob_key: []
        }
        self.softmax = nn.Softmax(dim=0)

    def test(self):
        self._test()
        self._save_output2excel('xlsx')
        
    def _test(self):
        """
        Test logic
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, load_data in enumerate(tqdm(self.data_loader)):
                if len(load_data) == 3: data, target, path = load_data
                elif len(load_data) == 2: data, target, path = load_data, None
                else: raise Exception('The length of load_data should be 2 or 3.')
                
                batch_num = batch_idx + 1
                
                # 1. To move Torch to the GPU or CPU
                data, target = data.to(self.device), target.to(self.device)

                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                logit, predict = torch.max(output, 1)
                if check_onehot_encoding_1(target[0].cpu(), self.classes): target = torch.max(target, 1)[-1] # indices
                
                # 3. Update the value
                use_output, use_predict = output.detach().cpu(), predict.detach().cpu()
                if path is not None: 
                    if isinstance(path, type([tuple, np.array])): path = list(path)
                    basic_path = ['.'.join(str(p).split('/')[-1].split('.')[:-1]) for p in path]
                    # Custom
                    basic_path = [p.split('_')[0] for p in basic_path]
                    # Custom
                    self.tracker[self.filename_key].extend(basic_path)
                elif self.filename_key in self.tracker.keys(): self.tracker.pop(self.filename_key)
                self.tracker[self.pred_key].extend([self.classes[pred] for pred in use_predict])
                self.tracker[self.prob_key].extend([self.softmax(el).tolist() for el in use_output])
        
    def _save_output2excel(self, excel_type:str='csv'):
        result = {
            self.filename_key: self.tracker[self.filename_key],
            self.pred_key: self.tracker[self.pred_key]
        }
        prob_vector = self.tracker[self.prob_key]

        prob_cnt = len(prob_vector[0]) if isinstance(prob_vector[0], type([list, np.array])) else 1
        if (prob_cnt == 1 and not isinstance(prob_vector[0], float)) or (prob_cnt != 1 and not isinstance(prob_vector[0][0], float)):
            raise TypeError('The probability must be a float type.')

        if prob_cnt != 1: 
            for idx in range(prob_cnt): result[f"{self.prob_key} - {self.classes[idx]}" if prob_cnt != 1 else self.prob_key] = np.array(prob_vector).T[idx]
        else: result[self.prob_key] = prob_vector
        
        result = pd.DataFrame(result)
        file_path = self.output_dir / 'result'
        if str(excel_type).lower() == 'xlsx': 
            result.to_excel(f'{str(file_path)}.xlsx', header=True, index=False, sheet_name='Result')
        else: 
            if str(excel_type).lower() != 'csv': 
                print(f'The Excel type is set to {excel_type}, but this is an unsupported format, so it is saved as CSV.')
                print('Please modify the function and rerun if you require this format.')
            result.to_csv(f'{str(file_path)}.csv', header=True, index=False)
        print(f'\nSave... {file_path}')        