import sys
import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

home_dir = './'
sys.path.append(home_dir)


class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop(self.fc1(out))
        out = self.fc2(out)
        out = self.fc3(out)
        return self.softmax(out)


class ServerSideNet(nn.Module):
    def __init__(self):
        super(ServerSideNet, self).__init__()
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.drop(self.fc1(x))
        out = self.fc2(out)
        out = self.fc3(out)
        return self.softmax(out)


class DeviceSideNet(nn.Module):
    def __init__(self):
        super(DeviceSideNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data = Variable(data.view(args.batch_size, 1, 28, 28))
        # target = Variable(target)
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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = Variable(data.view(args.test_batch_size, 1, 28, 28))
            # target = Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))


def FashionMNIST_training(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize((0.1307,),
                                                                                                       (0.3081,))])),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))])),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model = Net().to(device)
    model = FashionCNN().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'cnn_fashionmnist_fc.pt')


def normal_inference(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('\nNormal Inference Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                  100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def split_inference(server_model, device_model, device, test_loader):
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            device_output = device_model(data)
            print(device_output.shape)
            print(device_output.size(0))
            server_side_output = server_model(device_output)
            pred = server_side_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('\nSplit Inference Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FashionMNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current model')
    # parser.add_argument('--input_num_neurons', type=int, default=784, metavar='N',
    #                     help='number of neurons in the input layer')
    # parser.add_argument('--fc1_num_neurons', type=int, default=392, metavar='N',
    #                     help='number of neurons in the fc1 layer')
    parser.add_argument('--input_num_neurons', type=int, default=3072, metavar='N',
                        help='number of neurons in the input layer')
    parser.add_argument('--fc1_num_neurons', type=int, default=1536, metavar='N',
                        help='number of neurons in the fc1 layer')
    args = parser.parse_args()

    # FashionMNIST_training(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize((0.1307,),
                                                                                                       (0.3081,))])),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))])),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    entire_model_dict = torch.load('cnn_fashionmnist_fc.pt')
    print(entire_model_dict)
    test_model = FashionCNN().to(device)
    test_model.load_state_dict(entire_model_dict)
    normal_inference(test_model, device, test_loader)

    server_net = ServerSideNet().to(device)
    server_dict = server_net.state_dict()
    new_server_dict = {k: v for k, v in entire_model_dict.items() if k in server_dict.keys()}
    server_dict.update(new_server_dict)
    server_net.load_state_dict(server_dict)

    device_net = DeviceSideNet().to(device)
    device_dict = device_net.state_dict()
    new_device_dict = {k: v for k, v in entire_model_dict.items() if k in device_dict.keys()}
    device_dict.update(new_device_dict)
    device_net.load_state_dict(device_dict)

    split_inference(server_net, device_net, device, test_loader)
