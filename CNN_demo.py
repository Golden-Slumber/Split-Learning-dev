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
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *
from Hungarian_based_system_optimization import graph_based_alternating_optimization_framework, \
    subcarrier_aware_optimization, power_aware_optimization

home_dir = './'
sys.path.append(home_dir)


class CentralizedCNN(nn.Module):
    def __init__(self):
        super(CentralizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8 * 16 * 16, out_features=32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        self.max_pool2(out)

        out = out.view(x.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SplitCNN(nn.Module):
    def __init__(self, img_size_list):
        super(SplitCNN, self).__init__()
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

        # device-side sub-models
        self.sub_conv1s = nn.ModuleList()
        for n in range(self.n_devices):
            self.sub_conv1s.append(
                nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(4), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2)))
        # self.batch_norm = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)

        # cut layer
        self.cut_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(img_size_list[i] ** 2, 256, bias=False)) for i in range(self.n_devices)])
        # self.cut_layer_bias = None

        # server-side sub-model
        self.fc1 = nn.Linear(256, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.output = nn.Linear(64, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

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
                                                                                              self.tau2, self.P)
            elif self.mode == POWER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = power_aware_optimization(self.w_mat, self.h_mat,
                                                                                         self.tau2, self.P)
            elif self.mode == GRAPH:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = graph_based_alternating_optimization_framework(
                    self.w_mat, self.h_mat,
                    self.tau2, self.P)
            self.indicator_mat = None
            self.b_mat = None
            self.a_list = list()

            self.indicator_mat = torch.from_numpy(tmp_indicator_mat).to(self.device)
            self.b_mat = torch.from_numpy(tmp_b_mat).to(self.device)
            for j in range(256):
                self.a_list.append(torch.from_numpy(tmp_a_list[j]).to(self.device))
        return mse

    def forward(self, x):
        device_model_outputs = list()
        for n in range(self.n_devices):
            if n <= 1:
                device_input = x[:, :, int(32 * n):int(32 * n) + 32, :32]
            else:
                device_input = x[:, :, int(16 * (n - 2) % 64): int(16 * (n - 2) % 64) + 16,
                               16 * int((n - 2) / 4) + 32: 16 * int((n - 2) / 4) + 48]
            device_model_output = self.sub_conv1s[n](device_input)
            # device_model_output = self.batch_norm(device_model_output)
            # device_model_output = self.relu(device_model_output)
            # device_model_output = self.maxpool(device_model_output)
            device_model_output = device_model_output.view(-1, self.img_size_list[n] ** 2)
            device_model_outputs.append(device_model_output)

        for n in range(self.n_devices):
            device_model_outputs[n] = self.cut_layer[n](device_model_outputs[n])

        if self.sigma2 == 0:
            server_side_input = sum(device_model_outputs)
            res = self.relu(server_side_input)
            res = self.relu(self.fc1(res))
            res = self.relu(self.fc2(res))
            return self.softmax(self.output(res))
        else:
            tmp_h_mat = torch.from_numpy(self.h_mat).to(self.device)
            received_signal = torch.zeros((-1, 256)).to(self.device)
            for n in range(self.n_devices):
                transmit_signal = torch.multiply(device_model_outputs[n], self.b_mat[n])
                h_vec = torch.zeros(256).to(self.device)
                for j in range(256):
                    for k in range(self.indicator_mat.shape[1]):
                        if self.indicator_mat[j, k] == 1:
                            h_vec[j] = torch.sum(torch.mm(self.a_list[j].T, tmp_h_mat[n, k].reshape((5, 1))))
                received_signal += torch.multiply(transmit_signal, h_vec)
            noise = torch.normal(0, self.sigma2, (256, 5)).to(self.device)
            for j in range(256):
                received_signal[:, j] += torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
            res = self.relu(received_signal)
            res = self.relu(self.fc1(res))
            res = self.relu(self.fc2(res))
            return self.softmax(self.output(res))

class MyDataset(Dataset):

    def __init__(self, root, transform_status=True):
        self.root = root
        self.images_paths = [glob(f'{root}/{folder}/*.jpg') for folder in os.listdir(f"{root}")]
        self.images_paths = list(itertools.chain.from_iterable(self.images_paths))
        random.shuffle(self.images_paths)

        self.class_names = {class_name: label for label, class_name in enumerate(os.listdir(f"{root}"))}
        self.labels = [self.class_names[os.path.basename(os.path.dirname(path))] for path in self.images_paths]
        self.transform_status = transform_status
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image_path = self.images_paths[item]
        image = cv2.imread(image_path)[:, :, ::-1]
        image = cv2.resize(image, (64, 64))
        image = torch.tensor(image / 255).permute(2, 0, 1)
        if self.transform_status:
            image = self.transform(image)
        label = self.labels[item]

        return image.float(), torch.tensor([label])


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
    scheduler.step()


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


def centralized_model_training(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = CentralizedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            # print(data.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_set),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))


def EuroSAT_training(args, img_size_list):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    # data = MyDataset("./Resources/EuroSAT_RGB", transform_status=True)
    # train_set, test_set = train_test_split(data, test_size=0.2)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    # n_devices = 4
    # next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    # pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)
    model = SplitCNN(img_size_list).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'vertically_split_EuroSAT_n_devices_' + str(len(img_size_list)) + '.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='SGD weight decay (default:0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current model')
    args = parser.parse_args()

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
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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

    img_size_list = [32, 32, 16, 16, 16, 16, 16, 16, 16, 16]
    # EuroSAT_training(args, img_size_list)
