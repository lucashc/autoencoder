import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AE(nn.Module):
    """
    AutoEncoder for the alps
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                # Input is (N, 1, 512, 512)
                nn.Conv2d(1, 16, 3, padding=1), # Shape (N, 16, 512, 512)
                nn.Tanh(),
                nn.MaxPool2d(2, stride=2), # Shape (N, 16, 256, 256)
                nn.Conv2d(16, 32, 3, padding=1), # Shape (N, 32, 256, 256)
                nn.Tanh(),
                nn.MaxPool2d(2, stride=2), # Shape (N, 32, 128, 128)
                nn.Conv2d(32, 32, 3, padding=1), # Shape (N, 32, 128, 128)
                nn.Tanh(),
                nn.MaxPool2d(2, stride=2), # Shape (N, 32, 64, 64)
                nn.Conv2d(32, 16, 3, padding=1), # Shape (N, 16, 64, 64)
                nn.Tanh(),
                nn.MaxPool2d(2, stride=2) # Shape (N, 16, 32, 32)
            )
        self.decoder = nn.Sequential(
                # Transpose convolution operator
                nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1), # Shape (N, 32, 64, 64)
                nn.Tanh(),
                nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # Shape (N, 32, 128, 128)
                nn.Tanh(),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # Shape (N, 32, 256 256)
                nn.Tanh(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), # Shape (N, 32, 512, 512)  
                nn.ReLU()
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x