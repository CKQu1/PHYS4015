import torch
import torch.nn as nn
import copy 
import torch.nn.functional as F
import torch.optim as optim

class FullyConnected(nn.Module):

    def __init__(self, input_dim=28*28 , width=50, depth=3, num_classes=10):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x

def fc2(**kwargs):
    return FullyConnected(input_dim=28*28, width=100, depth=2, num_classes=10)

def fc3(**kwargs):
    return FullyConnected(input_dim=28*28, width=100, depth=3, num_classes=10)

def fc10(**kwargs):
    return FullyConnected(input_dim=28*28, width=100, depth=10, num_classes=10)

