import os
import random
import sys
import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import itertools
from glob import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import BasicBlock
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *
from Hungarian_based_system_optimization import graph_based_alternating_optimization_framework, \
    subcarrier_aware_optimization, power_aware_optimization

home_dir = './'
sys.path.append(home_dir)


class FullResnet(nn.Module):
    def __init__(self):
        super(FullResnet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        block = BasicBlock
        self.layer1 = self._make_layers(block, 16, 1, 1)
        self.layer2 = self._make_layers(block, 32, 2, 2)
        self.layer3 = self._make_layers(block, 64, 2, 2)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def _make_layers(self, block, planes, n_block, stride=1):
        layers = []
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
        layers.append(block(self.in_planes, planes, stride, downsample))
        for block_iter in range(1, n_block):
            layers.append(block(planes, planes))
        self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        return self.softmax(self.relu(self.fc(out)))


class RevisedSplitResnet(nn.Module):
    def __init__(self, img_size_list, device=None):
        super(RevisedSplitResnet, self).__init__()
        self.n_devices = len(img_size_list)
        self.img_size_list = img_size_list
        self.sigma2 = 0
        self.w_mat = None
        self.h_mat = None
        self.P = None
        self.indicator_mat = None
        self.b_mat = None
        self.a_list = list()
        self.mode = None
        self.device = device

        # device-side sub-models
        self.sub_models = nn.ModuleList()
        for n in range(self.n_devices):
            self.sub_models.append(
                nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(16), nn.ReLU(inplace=True)))
            block = BasicBlock
            self.sub_models.append(self._make_layers(block, 16, 16, 1, 1))
            self.sub_models.append(self._make_layers(block, 16, 32, 2, 2))
            self.sub_models.append(self._make_layers(block, 32, 64, 2, 2))
            self.sub_models.append(nn.AvgPool2d(int(self.img_size_list[n] / 4), stride=1))

        # cut layer
        self.cut_layers = nn.ModuleList()
        for n in range(self.n_devices):
            self.cut_layers.append(nn.Linear(64, 64, bias=False))

        # server-side model
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def _make_layers(self, block, in_planes, out_planes, n_block, stride=1):
        layers = []
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_planes * block.expansion))
        layers.append(block(in_planes, out_planes, stride, downsample))
        for block_iter in range(1, n_block):
            layers.append(block(out_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            # if n <= 1:
            #     device_model_output = x[:, :, int(16 * n):int(16 * n) + 16, :16]
            # else:
            #     device_model_output = x[:, :, int((8 * (n - 2)) % 32): int((8 * (n - 2)) % 32) + 8,
            #                           8 * int((n - 2) / 4) + 16: 8 * int((n - 2) / 4) + 24]
            if n <= 1:
                device_model_output = x[:, :, int(32 * n):int(32 * n) + 32, :32]
            else:
                device_model_output = x[:, :, int(16 * (n - 2) % 64): int(16 * (n - 2) % 64) + 16,
                               16 * int((n - 2) / 4) + 32: 16 * int((n - 2) / 4) + 48]
            for i in range(5):
                # print(device_model_output.shape)
                device_model_output = self.sub_models[5 * n + i](device_model_output)
            device_model_output = device_model_output.view(x.size(0), -1)
            cur_mean = device_model_output.mean()
            var += (device_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()
            device_model_outputs.append(device_model_output)
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
            device_model_outputs[n] = self.cut_layers[n](device_model_outputs[n])

        if self.sigma2 == 0:
            server_side_input = sum(device_model_outputs)
            # res = self.layer1(server_side_input.view(-1, 4, 8, 8))
            res = self.relu(server_side_input)
            res = self.dropout(res)
            res = self.relu(self.fc1(res))
            res = self.dropout(res)
            res = self.relu(self.fc2(res))
            # res = self.relu(self.fc2(res))
            return self.softmax(res)
        else:
            tmp_h_mat = torch.from_numpy(self.h_mat).to(self.device)
            received_signal = torch.zeros((1800, 64)).to(self.device)
            for n in range(self.n_devices):
                transmit_signal = torch.multiply(device_model_outputs[n], self.b_mat[n])
                h_vec = torch.zeros(64).to(self.device)
                for j in range(64):
                    for k in range(self.indicator_mat.shape[1]):
                        if self.indicator_mat[j, k] == 1:
                            h_vec[j] = torch.sum(torch.mm(self.a_list[j].T, tmp_h_mat[n, k].reshape((5, 1))))
                received_signal += torch.multiply(transmit_signal, h_vec)
            noise = torch.normal(0, self.sigma2, (64, 5)).to(self.device)
            for j in range(64):
                received_signal[:, j] += torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
            res = self.relu(received_signal)
            res = self.relu(self.fc1(res))
            res = self.relu(self.fc2(res))
            return self.softmax(res)

    def set_system_params(self, w_mat, h_mat, sigma2, P, mode):
        self.w_mat = w_mat
        self.h_mat = h_mat
        self.sigma2 = sigma2
        self.P = P
        self.mode = mode
        mse = 0
        if self.mode != PURE:
            if self.mode == SUBCARRIER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = subcarrier_aware_optimization(self.w_mat, self.h_mat,
                                                                                              self.sigma2, self.P,
                                                                                              max_iter=1)
            elif self.mode == POWER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = power_aware_optimization(self.w_mat, self.h_mat,
                                                                                         self.sigma2, self.P,
                                                                                         max_iter=20)
            elif self.mode == GRAPH:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = graph_based_alternating_optimization_framework(
                    self.w_mat, self.h_mat,
                    self.sigma2, self.P, max_iter=20)
            self.indicator_mat = None
            self.b_mat = None
            self.a_list = list()

            self.indicator_mat = torch.from_numpy(tmp_indicator_mat).to(self.device)
            self.b_mat = torch.from_numpy(tmp_b_mat).to(self.device)
            for j in range(64):
                self.a_list.append(torch.from_numpy(tmp_a_list[j]).to(self.device))
        return mse

    def get_statistics(self, x, J):
        variance_mat = numpy.zeros((self.n_devices, J))
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            # if n <= 1:
            #     device_model_output = x[:, :, int(16 * n):int(16 * n) + 16, :16]
            # else:
            #     device_model_output = x[:, :, int((8 * (n - 2)) % 32): int((8 * (n - 2)) % 32) + 8,
            #                           8 * int((n - 2) / 4) + 16: 8 * int((n - 2) / 4) + 24]
            if n <= 1:
                device_model_output = x[:, :, int(32 * n):int(32 * n) + 32, :32]
            else:
                device_model_output = x[:, :, int(16 * (n - 2) % 64): int(16 * (n - 2) % 64) + 16,
                               16 * int((n - 2) / 4) + 32: 16 * int((n - 2) / 4) + 48]
            for i in range(5):
                device_model_output = self.sub_models[5 * n + i](device_model_output)
            device_model_output = device_model_output.view(x.size(0), -1)
            cur_mean = device_model_output.mean()
            var += (device_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()
            device_model_outputs.append(device_model_output)
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
            device_model_outputs[n] = self.cut_layers[n](device_model_outputs[n])
            variance_mat[n, :] = torch.var(device_model_outputs[n], 0, correction=0).cpu().numpy()
        return variance_mat


class SplitResnet(nn.Module):
    def __init__(self, img_size_list, device=None):
        super(SplitResnet, self).__init__()
        self.n_devices = len(img_size_list)
        self.img_size_list = img_size_list
        self.sigma2 = 0
        self.w_mat = None
        self.h_mat = None
        self.P = None
        self.indicator_mat = None
        self.b_mat = None
        self.a_list = list()
        self.mode = None
        self.device = device
        self.in_planes = 16

        # device-side sub-models
        self.sub_conv1s = nn.ModuleList()
        for n in range(self.n_devices):
            self.sub_conv1s.append(
                nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(16), nn.ReLU(inplace=True)))
        # self.batch_norm = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.dropout = nn.Dropout(0.5)

        # cut layer
        self.cut_layers = nn.ModuleList()
        for i in range(self.n_devices):
            for p_iter in range(16):
                self.cut_layers.append(
                    nn.Sequential(nn.Linear(self.img_size_list[i] ** 2, 64, bias=False)))
        # self.cut_layers = nn.ModuleList()
        # for i in range(self.n_devices):
        #     self.cut_layers.append(nn.Sequential(nn.Linear(4 * self.img_size_list[i] ** 2, 256, bias=False)))
        # self.cut_layer_bias = None

        # server-side sub-model
        block = BasicBlock
        self.layer1 = self._make_layers(block, 16, 1, 1)
        self.layer2 = self._make_layers(block, 32, 2, 2)
        self.layer3 = self._make_layers(block, 64, 2, 2)

        self.fh = 2
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(64 * block.expansion, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def _make_layers(self, block, planes, n_block, stride=1):
        layers = []
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
        layers.append(block(self.in_planes, planes, stride, downsample))
        for block_iter in range(1, n_block):
            layers.append(block(planes, planes))
        self.in_planes = planes

        return nn.Sequential(*layers)

    def set_system_params(self, w_mat, h_mat, sigma2, P, mode):
        self.w_mat = w_mat
        self.h_mat = h_mat
        self.sigma2 = sigma2
        self.P = P
        self.mode = mode
        mse = 0
        if self.mode != PURE:
            if self.mode == SUBCARRIER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = subcarrier_aware_optimization(self.w_mat, self.h_mat,
                                                                                              self.sigma2, self.P,
                                                                                              max_iter=20)
            elif self.mode == POWER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = power_aware_optimization(self.w_mat, self.h_mat,
                                                                                         self.sigma2, self.P,
                                                                                         max_iter=20)
            elif self.mode == GRAPH:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = graph_based_alternating_optimization_framework(
                    self.w_mat, self.h_mat,
                    self.sigma2, self.P, max_iter=20)
            self.indicator_mat = None
            self.b_mat = None
            self.a_list = list()

            self.indicator_mat = torch.from_numpy(tmp_indicator_mat).to(self.device)
            self.b_mat = torch.from_numpy(tmp_b_mat).to(self.device)
            for j in range(64):
                self.a_list.append(torch.from_numpy(tmp_a_list[j]).to(self.device))
        return mse

    def get_statistics(self, x, J):
        variance_mat = numpy.zeros((self.n_devices, J))
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            if n <= 1:
                device_input = x[:, :, int(16 * n):int(16 * n) + 16, :16]
            else:
                device_input = x[:, :, int((8 * (n - 2)) % 32): int((8 * (n - 2)) % 32) + 8,
                               8 * int((n - 2) / 4) + 16: 8 * int((n - 2) / 4) + 24]
            device_model_output = self.sub_conv1s[n](device_input)
            # device_model_output = self.batch_norm(device_model_output)
            # device_model_output = self.relu(device_model_output)
            # device_model_output = self.maxpool(device_model_output)
            device_model_output = device_model_output.view(-1, 16 * (self.img_size_list[n] ** 2))
            cur_mean = device_model_output.mean()
            var += (device_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()
            device_model_outputs.append(device_model_output)
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
            device_model_outputs[n] = device_model_outputs[n].view(-1, 16, self.img_size_list[n] ** 2)
            for p_iter in range(16):
                device_model_outputs[n][:, p_iter, :] = self.cut_layers[16 * n + p_iter](
                    device_model_outputs[n][:, p_iter, :])

            variance_mat[n, :] = torch.var(device_model_outputs[n], 0, correction=0).cpu().numpy()
        return variance_mat

    def forward(self, x):
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            if n <= 1:
                device_input = x[:, :, int(16 * n):int(16 * n) + 16, :16]
            else:
                device_input = x[:, :, int((8 * (n - 2)) % 32): int((8 * (n - 2)) % 32) + 8,
                               8 * int((n - 2) / 4) + 16: 8 * int((n - 2) / 4) + 24]
            # if n <= 1:
            #     device_input = x[:, :, int(32 * n):int(32 * n) + 32, :32]
            # else:
            #     device_input = x[:, :, int(16 * (n - 2) % 64): int(16 * (n - 2) % 64) + 16,
            #                    16 * int((n - 2) / 4) + 32: 16 * int((n - 2) / 4) + 48]
            device_model_output = self.sub_conv1s[n](device_input)
            # print(device_model_output.shape)
            device_model_output = device_model_output.view(x.size(0), 16 * (self.img_size_list[n] ** 2))
            cur_mean = device_model_output.mean()
            var += (device_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()
            device_model_outputs.append(device_model_output)
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
            device_model_outputs[n] = device_model_outputs[n].view(x.size(0), 16, -1)
            tmp = torch.zeros((x.size(0), 16, 64)).to(self.device)
            for p_iter in range(16):
                tmp[:, p_iter, :] = self.cut_layers[16 * n + p_iter](
                    device_model_outputs[n][:, p_iter, :])
            device_model_outputs[n] = tmp
            # device_model_outputs[n] = self.cut_layers[n](device_model_outputs[n])

        if self.sigma2 == 0:
            server_side_input = sum(device_model_outputs)
            # res = self.layer1(server_side_input.view(-1, 4, 8, 8))
            res = self.relu(server_side_input.view(-1, 16, 8, 8))
            res = self.layer1(res)
            res = self.layer2(res)
            res = self.layer3(res)
            res = self.avgpool(res)
            res = res.view(res.size(0), -1)
            res = self.fc(res)
            # res = self.relu(self.fc2(res))
            return self.softmax(self.relu(res))
        else:
            res = torch.zeros((1000, 16, 64)).to(self.device)
            for p_iter in range(16):
                tmp_h_mat = torch.from_numpy(self.h_mat).to(self.device)
                received_signal = torch.zeros((1000, 64)).to(self.device)
                for n in range(self.n_devices):
                    transmit_signal = torch.multiply(device_model_outputs[n][:, p_iter, :].view(1000, -1),
                                                     self.b_mat[n])
                    h_vec = torch.zeros(64).to(self.device)
                    for j in range(64):
                        for k in range(self.indicator_mat.shape[1]):
                            if self.indicator_mat[j, k] == 1:
                                h_vec[j] = torch.sum(torch.mm(self.a_list[j].T, tmp_h_mat[n, k].reshape((5, 1))))
                    received_signal += torch.multiply(transmit_signal, h_vec)
                noise = torch.normal(0, self.sigma2, (64, 5)).to(self.device)
                for j in range(64):
                    received_signal[:, j] += torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
                res[:, p_iter, :] = received_signal.view(1000, 1, 64)

            res = self.layer1(res.view(-1, 16, 8, 8))
            res = self.layer2(res)
            res = self.layer3(res)
            res = self.avgpool(res)
            res = res.view(res.size(0), -1)
            res = self.fc(res)
            # res = self.relu(self.fc2(res))
            return self.softmax(res)


def tau_mat_processing(args, model, n_devices, J):
    tau_mat = numpy.zeros((n_devices, J))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # train_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=True, download=False,
    #                                            transform=transforms.Compose(
    #                                                [transforms.Pad(4), transforms.RandomHorizontalFlip(),
    #                                                 transforms.RandomCrop(32), transforms.ToTensor(),
    #                                                 transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                      (0.5, 0.5, 0.5))])),
    # train_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=True, download=False,
    #                                            transform=transforms.Compose(
    #                                                [transforms.ToTensor(),
    #                                                 transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                      (0.5, 0.5, 0.5))])),
    #                           batch_size=args.batch_size, shuffle=True, **kwargs)
    # with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
    #     train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    n_batches = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            partial_tau_mat = model.get_statistics(data, J)
            tau_mat += partial_tau_mat
            n_batches += 1
    return tau_mat / n_batches


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(-1)
        # print(data.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        # scheduler.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))
    return correct / len(test_loader.dataset)


def ResNet_training(args, img_size_list):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # train_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=True, download=False,
    #                                            transform=transforms.Compose(
    #                                                [transforms.ToTensor(),
    #                                                 transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                      (0.5, 0.5, 0.5))])),
    #                           batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=False,
    #                                           transform=transforms.Compose(
    #                                               [transforms.ToTensor(),
    #                                                transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                     (0.5, 0.5, 0.5))])),
    #                          batch_size=args.test_batch_size, shuffle=True, **kwargs)
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    # n_devices = 4
    # next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    # pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)
    # model = SplitResnet(img_size_list, device=device).to(device)
    # model = FullResnet().to(device)
    model = RevisedSplitResnet(img_size_list, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test(model, device, test_loader)

    if args.save_model:
        # torch.save(model.state_dict(), 'vertically_split_CIFAR10_n_devices_' + str(len(img_size_list)) + '.pt')
        torch.save(model.state_dict(), 'vertically_split_Resnet_n_devices_' + str(len(img_size_list)) + '.pt')


def plot_results(results, objectives, tau2_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(tau2_list, numpy.median(results[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(tau2_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Noise Variance', fontsize=25)
    plt.ylabel('Inference Accuracy', fontsize=25)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/ResNet_demo_' + data_name + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(tau2_list, numpy.median(objectives[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(tau2_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Noise Variance', fontsize=25)
    plt.ylabel('MSE', fontsize=25)
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/ResNet_demo_' + data_name + '_objective.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1800, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.7, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=0.002, metavar='M',
                        help='SGD weight decay (default:0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # dataset construction
    # data = MyDataset("./Resources/EuroSAT_RGB", transform_status=True)
    # train_set, test_set = train_test_split(data, test_size=0.2)
    # # print(train_set.shape)
    # # print(test_set.shape)
    # with open('./Resources/EuroSAT_train_set.pkl', 'wb') as f:
    #     pickle.dump(train_set, f)
    # with open('./Resources/EuroSAT_test_set.pkl', 'wb') as f:
    #     pickle.dump(test_set, f)
    # out_file_name = home_dir + 'Resources/EuroSAT_RGB_train_set.npz'
    # numpy.savez(out_file_name, train_set)

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    # device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # data = MyDataset("./Resources/EuroSAT_RGB", transform_status=True)
    # image, label = data[0]
    # print(len(data))
    # print(image[0].shape)
    # train_set, test_set = train_test_split(data, test_size=0.2)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # image, label = data[0]
    # print(image[0].shape)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(batch_idx)
    #     print(data.shape)

    # img_size_list = [16, 16, 8, 8, 8, 8, 8, 8, 8, 8]
    img_size_list = [32, 32, 16, 16, 16, 16, 16, 16, 16, 16]

    train_flag = False
    if train_flag:
        ResNet_training(args, img_size_list)
    else:
        # test_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=False,
        #                                           transform=transforms.Compose(
        #                                               [transforms.ToTensor(),
        #                                                transforms.Normalize((0.5, 0.5, 0.5),
        #                                                                     (0.5, 0.5, 0.5))])),
        #                          batch_size=args.test_batch_size, shuffle=True, **kwargs)
        with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
            test_set = pickle.load(f)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = RevisedSplitResnet(img_size_list, device=device).to(device)
        # model_state_dict = torch.load('vertically_split_CIFAR10_n_devices_' + str(len(img_size_list)) + '.pt')
        model_state_dict = torch.load('vertically_split_Resnet_n_devices_' + str(len(img_size_list)) + '.pt')
        model.load_state_dict(model_state_dict)

        # system parameters
        n_devices = len(img_size_list)
        J = 64
        K = 64
        m = 5
        P = 1
        tau2_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        # tau2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
        # tau2_list = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
        # tau2_list = [1, 2, 3, 4, 5, 6, 7, 8]
        # tau2_list = [1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
        # tau2_list = [0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4]
        # tau2_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        # tau2_list = [1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4]
        # tau2_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # ini_h_mat = abs(numpy.random.randn(n_devices, K, m))
        ini_h_mat = numpy.random.rayleigh(1, size=(n_devices, K, m))
        w_mat = tau_mat_processing(args, model, n_devices, J)
        # print(w_mat)

        repeat = 100
        # data_name = 'CIFAR10'
        data_name = 'EuroSAT'
        legends = ['Scheme 1: Proposed Approach', 'Scheme 2: Subcarrier-Aware', 'Scheme 3: Power-Aware']
        results = numpy.zeros((3, repeat, len(tau2_list)))
        objectives = numpy.zeros((3, repeat, len(tau2_list)))
        stored_results = numpy.zeros((3, len(tau2_list)))
        stored_objectives = numpy.zeros((3, len(tau2_list)))

        for r in range(repeat):
            h_mat = ini_h_mat.copy()
            h_norm_vec = numpy.zeros(K)
            for n in range(n_devices):
                j_idx = numpy.argmax(w_mat[n])
                subcarrier_scale_list = numpy.zeros(K)
                subcarrier_scale_list[0:int(K / 4)] = 0.01 * numpy.random.random_sample(int(K / 4)) + 0.01
                # subcarrier_scale_list[0:int(K / 4)] = 1
                subcarrier_scale_list[int(K / 4):int(K / 2)] = 0.2 * numpy.random.random_sample(int(K / 4)) + 0.2
                subcarrier_scale_list[int(K / 2):3 * int(K / 4)] = 1
                subcarrier_scale_list[3 * int(K / 4):] = 1
                subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
                for k in range(K):
                    h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]
                    h_norm_vec[k] = numpy.linalg.norm(h_mat[n, k])
                k_idx = numpy.argmin(h_norm_vec)
                tmp = h_mat[n, j_idx]
                h_mat[n, j_idx] = h_mat[n, j_idx]
                h_mat[n, k_idx] = tmp

            for i in range(len(tau2_list)):
                print('iteration ' + str(r) + ' - noise variance: ' + str(tau2_list[i]))

                objectives[0, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[i], P, GRAPH)
                stored_objectives[0, i] = objectives[0, r, i]
                results[0, r, i] = test(model, device, test_loader)
                stored_results[0, i] = results[0, r, i]

                objectives[1, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[i], P, SUBCARRIER_AWARE)
                stored_objectives[1, i] = objectives[1, r, i]
                results[1, r, i] = test(model, device, test_loader)
                stored_results[1, i] = results[1, r, i]

                objectives[2, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[i], P, POWER_AWARE)
                stored_objectives[2, i] = objectives[2, r, i]
                results[2, r, i] = test(model, device, test_loader)
                stored_results[2, i] = results[2, r, i]

            out_file_name = home_dir + 'Outputs/ResNet_demo_' + data_name + '_nvar-range_' + str(
                tau2_list[0]) + '-' + str(tau2_list[-1]) + '_repeat_' + str(r) + '_results.npz'
            numpy.savez(out_file_name, res=stored_results, obj=stored_objectives)
        plot_results(results, objectives, tau2_list, data_name, legends)
