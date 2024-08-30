from torch.utils.data.sampler import Sampler
from abc import abstractmethod

class BaseSampler(Sampler):
    def __init__(self, data_source , batch_size):
        # build data for sampling here
        self.batch_size = batch_size
        self.data = data_source 
        
    @abstractmethod
    def __iter__(self):
        """
        Implement logic of sampling

        :return: Model output
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)