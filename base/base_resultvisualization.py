import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import importlib
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy

class ResultVisualization:
    def __init__(self, parent_dir, result_name,
                 test_dirname:str='test', test_filename:str='metrics', test_file_addtional_name:str='test'):
        parent_dir = Path(parent_dir)
        self.name = result_name
        self.models_dir = parent_dir / 'models' / self.name
        self.output_dir = parent_dir / 'output' / self.name
        
        self.test_dirname = test_dirname
        self.test_filename = test_filename
        self.test_file_addtional_name = test_file_addtional_name
        self.sheet_list = ['training', 'validation', 'test']
        
        self.utils = self._import_custom_module()
        
        # output 중, metric.json 경로 정보 가져오기
        self.result_info = self._getMetricsFromConfig()
        # self.df_dict = self._json2df()

    def _import_custom_module(self, use_custom_module:str='utils'):
        if importlib.util.find_spec(use_custom_module) is None: 
            raise ModuleNotFoundError(f"The {module_name} module does not exist.")
        return importlib.import_module(use_custom_module)
    
    def _getMetricsFromConfig(self):
        config_name = 'config.json'
        config_paths = sorted(self.models_dir.glob(f'**/{config_name}'))
        if len(config_paths) == 0: raise ValueError(f'There is currently no config.json under the path ({self.models_dir}).')

        metircs_paths = OrderedDict()
        for config_path in config_paths:
            # Step 1. config.json를 통해 경로에 공통적으로 쓰이는 실험 경로를 가져옵니다.
            exper_name = '/'.join(self.utils.set_common_experiment_name(self.utils.read_json(config_path)).split('/')[1:])
            output_path = self.output_dir / exper_name

            # Step 2. 훈련에 사용된 metrics.json 파일을 찾습니다.
            training_metrics_path = sorted(output_path.glob('**/training/**/metrics.json')) 
            if len(training_metrics_path) == 1: training_metrics_path = training_metrics_path[-1]
            else: raise ValueError('The JSON file containing the training results could not be found.')
            
            # Step 3. 훈련이 멈춘 epoch을 찾습니다.
            latest_txt = sorted(config_path.parent.glob('latest.txt'))
            if len(latest_txt) == 1: 
                with open(latest_txt[-1], "r") as f:
                    latest = f.readlines()[-1].strip().split('epoch')[-1]
                if not latest.isnumeric(): raise ValueError(f'Warring: The latest epoch is unknown. -> {latest}')
                latest = int(latest)
            else: raise ValueError('The txt file containing the training epoch results could not be found.')

            # Step 4. 테스트에 사용된 metrics.json 파일을 찾습니다.
            test_metrics_path = sorted(output_path.glob(f'**/{self.test_dirname}/**/*{self.test_filename}*.json'))           
            if len(test_metrics_path) >= 1:
                test_epochs = []
                for t in test_metrics_path:
                    test_epoch = str(t).split('epoch')[-1].split('/')[0]
                    if not test_epoch.isnumeric(): raise ValueError(f'Warring: The testing epoch is unknown. -> {t.name}')
                    test_epochs.append(int(test_epoch))
                use_test_metrics_path = OrderedDict()
                for sort_index in np.argsort(test_epochs):
                    use_test_metrics_path[test_metrics_path[sort_index]] = test_epochs[sort_index]
            else:
                print(f'The JSON file containing the test results could not be found. -> {exper_name}')
                continue
                # raise ValueError('The JSON file containing the test results could not be found.')
            
            # Step 5. run id를 찾아, 딕셔너리에 업데이트합니다.
            run_id = str(training_metrics_path).split(exper_name)[-1].split('/')[1]
            metircs_paths[exper_name] = {run_id:{'config':config_path, 'train':training_metrics_path,'test':use_test_metrics_path, 'latest': latest}}
        
        # for idx, (exper_name, exper_result) in enumerate(metircs_paths.items()):
        #     print(f'{idx}. {exper_name} -> ({len(exper_result)} runs)')
        for run_id, run_json in metircs_paths[list(metircs_paths.keys())[0]].items():
            print(f'example run id: {run_id}')
            print(f'- latest epoch : {run_json["latest"]}')
            print(f'- train json file : {run_json["train"]}')
            for test_epoch, test_json in run_json['test'].items():
                print(f'- test json file : {test_epoch} -> {test_json}')
        return metircs_paths  
    
    def _json2df(self, base_date:tuple=(2024, 1, 1)):
        # 가져온 metric.json 기반으로 정보를 dataframe으로 변환 (테스트는 설정한 best_metric에 맞춰 높은 점수로 결정)
        sheet_data = self._set_a_df_info()
        df_dict = {s:'None' for s in self.sheet_list}
        
        for sheet, d_c in sheet_data.items():
            df = pd.DataFrame(data=d_c['data'], columns=d_c['col'][0])
            model_mean_time = []
            if 'Runtime' in df.columns:
                df.loc[:, 'Runtime'] = pd.to_datetime([self.utils.convert_to_datetime(self.utils.convert_days_to_hours(t), base_date) for t in df.loc[:, 'Runtime'].tolist()])
                for model_name in df['Model'].unique():
                    mean_datetime = pd.to_datetime(df[df['Model']==model_name]['Runtime'].apply(lambda x: x.timestamp()).mean(), unit='s')
                    elapsed_time_str = self.utils.format_elapsed_time(mean_datetime, base_date) # HH:MM:SS
                    elapsed_time_min  = int(elapsed_time_str.split(':')[0]) * 60 + int(elapsed_time_str.split(':')[1])
                    model_mean_time.append([model_name, elapsed_time_str, elapsed_time_min])
                df['Runtime'] = [self.utils.format_elapsed_time(t, base_date) for t in df.loc[:, 'Runtime'].tolist()] # HH:MM:SS # df['runtime'].time()
            df_dict[sheet] = deepcopy(df)
            
            if model_mean_time != []:
                time_df = pd.DataFrame(data=model_mean_time, columns=['Model', 'Runtime (mean)', 'Minutes (mean)']) 
                sheet_name =  str(sheet)+'_runtime'
                df_dict[sheet_name] = deepcopy(time_df)
        return df_dict   
    
    def _set_a_df_info(self):
        '''
        0. BASIC CONTENT: See `set_common_experiment_name` function in utils.config_util.py.
        1. SAVE CONTENT - training   : BASIC CONTENT | real_epoch | Runtime | metrics...
        2. SAVE CONTENT - validation : BASIC CONTENT | real_epoch | metrics...
        3. SAVE CONTENT - test       : BASIC CONTENT | real_epoch | metrics...
        '''
        basic_column_list = ['Run ID', 
                             'Model', 'DataLoader',
                             'Optimizer', 'Loss', 
                             'Learning rate (LR)', 'LR scheduler',
                             'DA', 'Sampler type', 'Sampler',
                             'Batch size', 'Accumulation steps', 
                             'Max Epoch', 'Last Training Epoch']
        data_dict = {s:{'data':[], 'col':[]} for s in self.sheet_list}
        
        for exper_name, exper_dict in self.result_info.items():
            for run_id, run_json in exper_dict.items():
                # 0. get a config
                exper_dict = self.utils.set_common_experiment_name(self.utils.read_json(run_json['config']), return_type=dict)
                basic_data = [run_id, 
                              exper_dict['model'], exper_dict['dataloader'],
                              exper_dict['optimizer'], exper_dict['loss'],
                              exper_dict['lr'], exper_dict['lr_scheduler'],
                              exper_dict['da'], exper_dict['sampler_type'], exper_dict['sampler'],
                              exper_dict['batch_size'], exper_dict['accum_steps'], 
                              exper_dict['max_epoch'], run_json['latest']]
                # 1. get a test information
                test_data, test_col, test_epoch = self._get_test_result(run_json['test'], basic_data, basic_column_list)
                data_dict['test']['data'].append(test_data)
                data_dict['test']['col'].append(test_col)
                # 2. get a train information
                tr, val, _ = self._read_df_result(run_json['train'], test_epoch, mode='train')
                data_dict['training']['data'].append(self._list_concat(basic_data, list(tr.values())))
                data_dict['training']['col'].append(self._list_concat(basic_column_list, list(tr.keys())))
                if len(list(val.keys())) != 1:
                    data_dict['validation']['data'].append(self._list_concat(basic_data, list(val.values())))
                    data_dict['validation']['col'].append(self._list_concat(basic_column_list, list(val.keys())))
        return data_dict
    
    def _get_test_result(self, run_json, basic_data, basic_column_list):
        # return data, col, test_epoch
        raise NotImplementedError
        
    def _read_df_result(self, json_path, test_epoch, mode='train'):
        if mode not in ['train', 'test']: TypeError('The model can only accept "train" and "test" as inputs.')
        json_content = self.utils.read_json(json_path)
        test_epoch_key = 'Best Epoch'
        train, valid, test = {test_epoch_key:test_epoch}, {test_epoch_key:test_epoch}, {test_epoch_key:test_epoch}
        if mode == 'train': 
            train = {test_epoch_key:test_epoch, 'Runtime':json_content['totaltime'] if 'totaltime' in json_content.keys() else '00:00:00'}
        for k, v in json_content.items():
            if any(s in k for s in ['epoch', 'time', 'confusion']): continue
            if mode == 'train':
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if 'val' in kk: valid[f'{k}_{kk.split("val_")[-1]}'] = vv[test_epoch-1] # epoch strat from 1 but index start from 0
                        else: train[f'{k}_{kk}'] = vv[test_epoch-1]
                else:
                    if 'val' in k: valid[k.split('val_')[-1]] = v[test_epoch-1]
                    else: train[k] = v[test_epoch-1]
            else: 
                if isinstance(v, dict):
                    for kk, vv in v.items(): test[f'{k}_{kk}'] = vv
                else: test[k] = v
        return train, valid, test

    def _list_concat(self, one, two):
        data = deepcopy(one)
        data.extend(two)
        return data
    
    # 변환한 dataframe을 기반으로 시각화 출력물 저장하기
    def _close_all_plots(self):    
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
    def save2excel(self, save_name:str=None, extension:str='xlsx'):
        # 0. SAVE FILE : self.output_dir / (self.name.extension or save_name.extension)
        save_name = save_name if save_name is not None else self.name
        save_name += f'.{extension}' 
        if str(save_name)[0] != '/': save_path = self.output_dir / save_name
        else: save_path = Path(save_name)
        if not save_path.parent.is_dir(): self.utils.ensure_dir(save_path.parent, True)
        
        writer=pd.ExcelWriter(save_path, engine='openpyxl')
        for sheet_name, df in self.df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
        writer.close()
        print(f'save... {save_name}')
    
    def save2bar_per_model(self, df, optimizer, loss, by_metric:str, by_metric_replace_str:str=None, by:list=['lr scheduler', 'DA', 'Sampling'], 
                           min_y:float=0.75, bar_width=0.8, figsize:tuple=(17, 5), colormap:str='tab20', xtick_rotation:int=0,
                           show_score:bool=True, show_legend:bool=True, show_result:bool=True,
                           save_name:str=None, extension:str='png', just_show:bool=False):
        # by에는 optimizer, loss를 제외한 구분 특성을 입력해야 합니다.
        save_name = save_name if save_name is not None else self.name
        save_name += f'.{extension}' 
        if str(save_name)[0] != '/': save_path = self.output_dir / save_name
        else: save_path = Path(save_name)
        if not save_path.parent.is_dir(): self.utils.ensure_dir(save_path.parent, True)
        
        title = (by_metric.lower().capitalize() if by_metric_replace_str is None else by_metric_replace_str) + ' by '
        for idx, b in enumerate(by, 1):
            title += b.lower().capitalize()
            if idx <= len(by)-2: title += ', '
            elif idx == len(by)-1: title += ' and '
            else: title += '.'
        
        index_list = ['model']
        index_list.extend(deepcopy(by))
        stack_list = deepcopy(by)
        choose_list = deepcopy(by)
        choose_list.append('model')
        choose_list.append(by_metric)
        
        need_info = df.loc[(df['optimizer']==optimizer) & (df['loss']==loss), choose_list]
        need_info = need_info.set_index(index_list)
        pivot_table = need_info[by_metric].unstack(stack_list)

        ax = pivot_table.plot(kind='bar', stacked=False, figsize=figsize, fontsize=10, colormap=colormap, ylim=[min_y, 1], width=bar_width)

        # 각 막대에 값 표시
        if show_score:
            for bar_container in ax.containers: # 막대 컨테이너 가져오기
                for bar, _ in zip(bar_container, pivot_table.columns):
                    height = bar.get_height()  # 막대의 높이 가져오기
                    padding = 0.05
                    val = str(round(height*100, 2)).replace('.00', '')
                    if val[-1] == '0' and '.' in val: val = val[:-2]
                    val = ''.join([h+'\n' for h in val])
                    if height-padding <= min_y: height = (min_y if height < min_y else height) + padding
                    ax.text(bar.get_x() + bar.get_width() / 2, height - padding, val, ha='center', va='bottom', fontsize=8)

        # 범례 설정
        if show_legend:
            ax.legend(title='Legend', loc='upper right', fontsize='small')  # 범례 위치 및 글꼴 크기 설정
            # 범례 항목 텍스트 변경
            new_labels = []
            for _ in pivot_table.columns:
                new_label = ''
                for idx, s in enumerate(stack_list):
                    new_value = 'None' if pd.isna(_[idx]) else _[idx]
                    new_label += f'{s} - {new_value}'
                    if idx < len(stack_list)-1: new_label += ' | '
                new_labels.append(new_label)
            ax.legend(labels=new_labels, loc='lower center', bbox_to_anchor=(0,-0.7,1,0.2), ncol=1)#, mode='expand')  # 범례 항목 텍스트 변경

        # x 축 눈금(label) 설정
        for tick in ax.get_xticklabels(): tick.set_rotation(xtick_rotation)

        # 그래프 제목 및 축 이름 설정
        fontsize, labelpad = 15, 15
        plt.title(f'\nOptim - {optimizer} & loss - {loss}', loc='right', fontsize=10)
        plt.title(title, fontsize=fontsize, pad=labelpad)
        plt.xlabel('Model', fontsize=fontsize, labelpad=labelpad)
        plt.ylabel(by_metric.lower().capitalize() if by_metric_replace_str is None else by_metric_replace_str, fontsize=fontsize, labelpad=labelpad)
        if not just_show: plt.savefig(save_path, bbox_inches="tight")
        else: return plt
        
        # 그래프 보여주기
        if show_result: plt.show()
        self._close_all_plots()
    # 변환한 dataframe을 기반으로 시각화 출력물 저장하기