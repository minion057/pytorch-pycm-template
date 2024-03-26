from base import BaseModel

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(BaseModel):
    def __init__(self, num_classes=10, H:int=28, W:int=28, in_chans:int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        OH, OW = self.pooling_output_size(self.conv_output_size(H, 5),2),  self.pooling_output_size(self.conv_output_size(W, 5),2)
        OH, OW = self.pooling_output_size(self.conv_output_size(OH, 5),2), self.pooling_output_size(self.conv_output_size(OW, 5),2)
        flatten_shape = OH * OW * 20
        self.fc1 = nn.Linear(flatten_shape, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flat(x) # x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
