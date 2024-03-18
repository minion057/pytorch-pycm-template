import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseRawDataLoader

class MnistDataLoader(BaseRawDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        classes = [0,1,2,3,4,5,6,7,8,9]
        super().__init__(self.dataset, classes, batch_size, shuffle, validation_split, num_workers)
