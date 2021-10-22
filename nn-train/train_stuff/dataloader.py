import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from typing import Tuple, List

from torchvision.transforms import functional as F

def get_data_loaders(ngpu, batch_size):

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])


    # no data agumentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if ngpu else {}

    # MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, **kwargs)
    return trainloader, testloader, trainset


