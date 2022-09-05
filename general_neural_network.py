import sys
import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

home_dir = './'
sys.path.append(home_dir)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 768, bias=True)
        self.fc2 = nn.Linear(768, 192, bias=True)
        self.fc3 = nn.Linear(196, 98, bias=True)
        self.fc4 = nn.Linear(98, 49, bias=True)
        self.output = nn.Linear(49, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.softmax(self.output(x))
