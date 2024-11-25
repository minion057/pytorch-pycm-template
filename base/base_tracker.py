import pandas as pd
import numpy as np
from pathlib import Path

class BaseTracker:
    def __init__(self, *keys, columns:list|np.ndarray):
        if not isinstance(columns, (list, np.array)): 
            raise ValueError('Columns should be a list or numpy.array.')
        elif isinstance(columns, np.ndarray): columns = columns.tolist()
        self._data = pd.DataFrame(index=keys, columns=columns)
        self.index = self._data.index.values
        self.columns = columns
        self.reset()

    def reset(self):
        for key in self.index:
            for c in self.columns:
                self._data.loc[key, c] = []

    def update_all(self, key, value:dict):
        if not isinstance(value, dict): raise ValueError('Value should be a dictionary.')
        if not all(k in list(value.keys()) for k in self.columns):
            raise ValueError('The value should contain all columns.')
        for c, v in value.items():
            self._data.loc[key, c].append(v)
        
    def update(self, key, column, value):
        if key not in self.index: raise ValueError(f'Key {key} is not found in index.')
        if column not in self.columns: raise ValueError(f'Column {column} is not found in columns.')
        self._data.loc[key, column].append(value)
            
    def get_data(self, key=None, column=None):
        if key is None and column is None: return self.result()
        elif key is None and column is not None: return {k:v for k, v in self._data.loc[:, column].items()}
        elif key is not None and column is None: return {k:v for k, v in self._data.loc[key, :].items()}
        else: return self._data.loc[key, column]
        
    def result(self):
        return {key:self.get_data(key)for key in self.index}

    def save2excel(self, savedir:str, savename:str='result', excel_type:str='csv'):
        file_path = Path(savedir) / savename
        if len(self.index) > 1 and excel_type == 'csv': 
            print(f'Cannot save data as CSV when each key requires a separate sheet. Please use XLSX format to save each key as a different sheet.')
            excel_type = 'xlsx'
        
        if str(excel_type).lower() == 'xlsx':
            with pd.ExcelWriter(f'{str(file_path)}.xlsx', engine='openpyxl') as writer:
                for key in self.index:
                    result = self.get_data(key)
                    max_item = max([len(v) for v in result.values()])
                    result = {k:(v if v != [] else ['']*max_item) for k,v in result.items()}
                    df = pd.DataFrame(result)
                    df.to_excel(writer, sheet_name=key, index=False, header=True)
        else: 
            if str(excel_type).lower() != 'csv': 
                print(f'The Excel type is set to {excel_type}, but this is an unsupported format, so it is saved as CSV.')
                print('Please modify the function and rerun if you require this format.')
            self._data.to_csv(f'{str(file_path)}.csv', index=True, header=True)
        print(f'\nSave... {file_path}.{excel_type}\n')
        
    