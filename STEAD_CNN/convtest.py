import torch

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from dataset import HDF5Dataset
from torch.utils.data import DataLoader

from model import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    #train_dataset = HDF5Dataset('/home/ph/PycharmProjects/STEAD_ANN/MiniTrain.hdf5')
    train_dataset = HDF5Dataset('../../PycharmProjects/STEAD_ANN/Train_data.hdf5')
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    trace, label = next(iter(trainloader))
    trace, label = next(iter(trainloader))

    print(label)
    print(trace)
    print(trace.shape)
    print(type(trace))
    print(type(trace.numpy()))

    plt.figure()
    plt.plot(trace.squeeze().numpy())
    plt.show()

    # eq, labels = next(iter(trainloader))
    #
    # print(f'input size: {eq.shape}')
    #
    # eq = torch.unsqueeze(eq, 0)
    #
    # print(f'input size: {eq.shape}')
    #
    # convl1 = nn.Conv1d(1, , 2, stride=2)
    #
    # eq = convl1(eq)
    #
    # print(f'output size: {eq.shape}')

    # data = torch.ones((5, 1, 6000))

    # net = ClassConv()

    # out = net(data)

    # print(out.shape)


if __name__ == "__main__":
    main()
