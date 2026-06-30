# %% [code]
# %% [code]
# %% [code]
import math 
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ConvBN2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.01):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Save the slope so it is accessible in the forward pass and init steps
        self.negative_slope = negative_slope 

        # Execute custom weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv.weight, a=self.negative_slope, nonlinearity='leaky_relu')
        
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

        init.ones_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def forward(self, x):
        # Using the saved instance variable self.negative_slope
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=self.negative_slope)


class PseudoAlexNet(nn.Module):
    def __init__(self, p=0.5, negative_slope=0.01, tiny_factor: int = 1, is_rgba: bool = False):
        super(PseudoAlexNet, self).__init__()
        # Input image dimension baseline: 3 x 300 x 200
        self.tiny_factor = tiny_factor
        # --- 1. THE VISION BACKBONE (All spatial layers go here) ---
        self.features = nn.Sequential(
            # Layer 1
            ConvBN2d(in_channels=3 + int(is_rgba), out_channels=int((1.0/self.tiny_factor)*12), kernel_size=5, stride=2, padding=2, negative_slope=negative_slope),
            
            # Layer 2 + Pool
            ConvBN2d(in_channels=int((1.0/self.tiny_factor)*12), out_channels=int((1.0/self.tiny_factor)*16), kernel_size=3, stride=1, padding=1, negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            ConvBN2d(in_channels=int((1.0/self.tiny_factor)*16), out_channels=int((1.0/self.tiny_factor)*24), kernel_size=3, stride=1, padding=1, negative_slope=negative_slope),
            
            # Layer 4 + Pool
            ConvBN2d(in_channels=int((1.0/self.tiny_factor)*24), out_channels=int((1.0/self.tiny_factor)*32), kernel_size=3, stride=1, padding=1, negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # --- 2. THE CLASSIFICATION HEAD (All vector dense layers go here) ---
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(int((1.0/self.tiny_factor)*32) * 37 * 25, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p),
            nn.Linear(256, 101)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=negative_slope, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(layer.bias, -bound, bound)

    def forward(self, x):
        # 1. Extract structural features [Batch, 32, 37, 25]
        x = self.features(x)
        
        # 2. Flatten spatial dimensions into a single vector
        x = x.view(x.size(0), -1) 

        # 3. Pass through the classification brain
        x = self.classifier(x)
        
        return x