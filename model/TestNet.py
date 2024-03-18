from torch import nn
from base import BaseModel

class TestNet(BaseModel):
    def __init__(self, num_classes:int, H:int, W:int):
        super().__init__()
        if num_classes <= 0 or H <= 0 or W <= 0: raise ValueError('The number of classes (num_classes), height (H) and width (W) must be positive.')
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
            
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),         
        )
        self.flat = nn.Flatten()
        OH, OW = self.pooling_output_size(self.conv_output_size(H, 3),2),  self.pooling_output_size(self.conv_output_size(W, 3),2)
        OH, OW = self.pooling_output_size(self.conv_output_size(OH, 3),2), self.pooling_output_size(self.conv_output_size(OW, 3),2)
        OH, OW = self.pooling_output_size(self.conv_output_size(OH, 3),2), self.pooling_output_size(self.conv_output_size(OW, 3),2)
        flatten_shape = OH * OW * 32
        self.fc_layer = nn.Sequential(
            nn.Linear(flatten_shape, 128),                                     
            nn.ReLU(),
            nn.Linear(128, num_classes)                                                   
        )   

    def forward(self, x):
        out = self.layer(x)
        out = self.flat(out)
        out = self.fc_layer(out)
        return out
