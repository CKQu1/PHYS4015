from __future__ import print_function
import torch
import os
import random
import numpy as np
import argparse
import scipy.io as sio
import math

import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import train_stuff.model_loader as model_loader
from train_stuff.dataloader import get_data_loaders

# way to extract the weight parameters
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training with save all transient state in one epoch
def train_save(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    sub_loss = []
    sub_weights = []

    # if you change your loss function type, you will have to modify the following accordingly
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_size = inputs.size(0)
        total += batch_size
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # record tiny steps in every epoch
        sub_loss.append(loss.item())


        train_loss += loss.item()*batch_size
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    # save weight at the end of epoch
    w = get_weights(net) # initial parameters
    for j in range(len(w)):
        w[j] = w[j].cpu().numpy()
    sub_weights.append(w)
  
    return train_loss/total, 100 - 100.*correct/total, sub_weights, sub_loss

def test_save(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sub_loss = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        batch_size = inputs.size(0)
        total += batch_size

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        sub_loss.append(loss.item())
        test_loss += loss.item()*batch_size
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total, sub_loss

def name_save_folder(model,epochs,lr,batch_size,ngpu,idx):
    save_folder = model + '_epoch' + str(epochs) + '_lr=' + str(lr)

    save_folder += '_bs=' + str(batch_size)
    
    if ngpu > 1:
        save_folder += '_ngpu=' + str(ngpu)
    if idx:
        save_folder += '_idx=' + str(idx)

    return save_folder

if __name__ == '__main__':

    # network type
    model = 'fc2'
    net = model_loader.load(model)

    # training hyperparameters
    batch_size = 1024
    lr = 0.1
    momentum = 0
    epochs = 5

    # number of GPUs to use
    ngpu = 1

    # the index for the repeated experiment
    idx = 0

    # type of loss function
    criterion = nn.CrossEntropyLoss()

    # starting epoch
    start_epoch = 1

    print('\nLearning Rate: %f' % lr)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))

    # Set the seed for reproducing the results
    rand_seed = 0
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(rand_seed)
        cudnn.benchmark = True

    save_folder = name_save_folder(model,epochs,lr,batch_size,ngpu,idx)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    #f = open('trained_nets/' + save_folder + '/log.out', 'a')

    # load MNIST dataset
    trainloader, testloader, _ = get_data_loaders(ngpu, batch_size)        

    # parallelization
    if ngpu > 1:
        net = torch.nn.DataParallel(net)

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=False)

    # training logs per iteration
    training_history = []
    testing_history = []
   
    for epoch in range(start_epoch, epochs + 1):
        print(epoch)        
        # Save checkpoint.

        loss, train_err, sub_weights, sub_loss = train_save(trainloader, net, criterion, optimizer, use_cuda)
        test_loss, test_err, test_sub_loss = test_save(testloader, net, criterion, use_cuda)

        # save loss and weights in each tiny step in every epoch
        sio.savemat('trained_nets/' + save_folder + '/model_' + str(epoch) + '_sub_loss_w.mat',
                            mdict={'sub_weights': sub_weights,'sub_loss': sub_loss, 'test_sub_loss': test_sub_loss},
                            )            

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)

        # validation acc
        acc = 100 - test_err

        # record training history (starts at initial point)
        training_history.append([loss, 100 - train_err])
        testing_history.append([test_loss, acc])

 
    sio.savemat('trained_nets/' + save_folder + '/' + model + '_loss_log.mat',
                        mdict={'training_history': training_history,'testing_history': testing_history},
                        )

