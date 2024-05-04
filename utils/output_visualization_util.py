from copy import deepcopy
from base.base_resultvisualization import ResultVisualization

class ResultBestVisualization(ResultVisualization):
    def __init__(self, parent_dir, result_name, test_dirname:str='test', test_filename:str='metrics',
                 best_metric=None, comparison_metric=None, threshold:float=0.5):
        if best_metric is None: raise ValueError('best_metric is required.')
        super().__init__(parent_dir, result_name, test_dirname, test_filename)
        self.best_metric = best_metric
        self.comparison_metric = comparison_metric
        self.threshold = threshold
        
        # output 중, metric.json 경로 정보 가져오기
        self.df_dict = self._json2df()
        
    def _get_test_result(self, run_json, basic_data, basic_column_list):
        best_result = None
        latest = run_json['latest'] if 'latest' in list(run_json.keys()) else None
        for test_epoch, test_json in run_json['test'].items():
            _, _, te = self._read_df_result(test_json, 'test', latest)
            if best_result is None:
                best_result = deepcopy(te)
                if self.best_metric not in list(te.keys()): raise ValueError(f'Not found {self.best_metric}')
                if self.comparison_metric is not None and self.comparison_metric not in list(te.keys()): raise ValueError(f'Not found {self.comparison_metric}')
            else:
                if self.comparison_metric is not None and abs(te[self.best_metric] - te[self.comparison_metric]) > self.threshold: continue
                if best_result[self.best_metric] < te[self.best_metric]: best_result = deepcopy(te)
        return self._list_concat(basic_data, list(best_result.values())), self._list_concat(basic_column_list, list(best_result.keys()))

class ResultFixedVisualization(ResultVisualization):
    def __init__(self, parent_dir, result_name, test_dirname:str='test', test_filename:str='metrics', test_file_addtional_name:str='test'):
        super().__init__(parent_dir, result_name, test_dirname, test_filename)
        self.test_file_addtional_name = test_file_addtional_name
        
        # output 중, metric.json 경로 정보 가져오기
        self.df_dict = self._json2df()
        
    def _get_test_result(self, run_json, basic_data, basic_column_list):
        test_epoch_list = list(run_json['test'].keys())
        if len(test_epoch_list): 
            _, _, te = self._read_df_result(run_json['test'][test_epoch_list[0]], 'test', test_epoch_list[0])
        else:
            for test_epoch, test_json in run_json['test'].items():
                if self.test_file_addtional_name in test_json:
                    _, _, te = self._read_df_result(test_json, 'test', test_epoch)
                    break
        return self._list_concat(basic_data, list(te.values())), self._list_concat(basic_column_list, list(te.keys()))
