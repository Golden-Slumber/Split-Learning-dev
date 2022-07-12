import sys
import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

home_dir = '../'
sys.path.append(home_dir)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1008)
        self.fc2 = nn.Linear(1008, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 500)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class DeviceNet(nn.Module):
    def __init__(self):
        super(DeviceNet, self).__init__()
        self.fc1 = nn.Linear(392, 504)

    def forward(self, x):
        x = x.view(-1, 392)
        x = F.relu(self.fc1(x))
        return x


class SingleDeviceNet(nn.Module):
    def __init__(self):
        super(SingleDeviceNet, self).__init__()
        self.fc1 = nn.Linear(784, 1008)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return x


class CutLayer(nn.Module):
    def __init__(self, tau2=0):
        super(CutLayer, self).__init__()
        self.tau2 = tau2
        self.fc2 = nn.Linear(1008, 500)

    def forward(self, x):
        x = x.view(-1, 1008)
        x = self.fc2(x)
        if self.tau2 != 0:
            noise = torch.normal(0, self.tau2, x.shape)
            x = x + noise
        x = F.relu(x)
        return x


class NoisyCutLayer(nn.Module):
    def __init__(self):
        super(NoisyCutLayer, self).__init__()
        self.tau2 = 0.1
        self.fc2 = nn.Linear(1008, 500)

    def forward(self, x):
        x = x.view(-1, 1008)
        noise = torch.normal(0, self.tau2, (x.shape[0], 500)).to(device)
        x = F.relu(self.fc2(x) + noise)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))


def normal_inference(model, device, test_loader):
    inference_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            inference_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    inference_loss /= len(test_loader.dataset)
    print('\nNormal Inference Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                  100. * correct / len(test_loader.dataset)))


def test_inference(server_model, cut_layer, device_model, device, test_loader):
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            device_side_output = device_model(data)
            cut_layer_output = cut_layer(device_side_output)
            server_side_output = server_model(cut_layer_output)
            pred = server_side_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('\nTest Inference Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                100. * correct / len(test_loader.dataset)))


def split_inference(server_model, cut_layer, device_models, device, test_loader):
    correct = 0
    n_devices = len(device_models)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            data = data.view(-1, 28 * 28)
            data_list = torch.chunk(data, chunks=n_devices, dim=1)
            device_side_output = torch.zeros((data.shape[0], 392))
            for i in range(n_devices):
                device_data = data_list[i]
                device_output = device_models[i](device_data)
                if i == 0:
                    device_side_output = device_output
                else:
                    device_side_output = torch.cat([device_side_output, device_output], dim=1)
            cut_layer_output = cut_layer(device_side_output)
            server_side_output = server_model(cut_layer_output)
            pred = server_side_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print('\nSplit Inference Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                                 100. * correct / len(test_loader.dataset)))


def FashionMNIST_training():
    parser = argparse.ArgumentParser(description='Fashion MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current model')
    args = parser.parse_args()

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

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'fashionmnist_fc.pt')


if __name__ == '__main__':
    # print('pyCharm')
    # model_dict = torch.load('mnist_fc.pt')
    # print(model_dict)
    # print(model_dict['fc1.weight'])
    # print(type(model_dict['fc1.weight']))
    # print(model_dict['fc1.weight'].shape)
    # print(model_dict['fc1.bias'].shape)

    parser = argparse.ArgumentParser(description='Fashion MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))])),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    entire_model_dict = torch.load('fashionmnist_fc.pt')
    # normal inference
    model = Net().to(device)
    model.load_state_dict(entire_model_dict)
    normal_inference(model, device, test_loader)

    # split inference
    server_model = ServerNet().to(device)
    n_devices = 2
    device_models = []
    for i in range(n_devices):
        device_model = DeviceNet().to(device)
        device_models.append(device_model)
    tau2 = 0
    cut_layer = CutLayer(tau2=tau2).to(device)

    # server-side model parameters set
    server_dict = server_model.state_dict()
    new_server_dict = {k: v for k, v in entire_model_dict.items() if k in server_dict.keys()}
    server_dict.update(new_server_dict)
    server_model.load_state_dict(server_dict)

    # cut layer parameters set
    cut_layer_dict = cut_layer.state_dict()
    new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
    cut_layer_dict.update(new_cut_layer_dict)
    cut_layer.load_state_dict(cut_layer_dict)

    # device-side model parameters set
    device_side_weights = entire_model_dict['fc1.weight']
    device_side_bias = entire_model_dict['fc1.bias']
    for i in range(n_devices):
        device_dict = device_models[i].state_dict()
        fc1_weight = torch.zeros((504, 392))
        fc1_bias = torch.zeros(504)
        for j in range(504):
            fc1_bias[j] = device_side_bias[i * n_devices + j]
            for k in range(392):
                fc1_weight[j, k] = device_side_weights[i * n_devices + j, i * n_devices + k]

        new_device_dict = {'fc1.weight': fc1_weight.to(device), 'fc1.bias': fc1_bias.to(device)}
        device_dict.update(new_device_dict)
        device_models[i].load_state_dict(device_dict)

    # test inference
    single_device_model = SingleDeviceNet().to(device)
    single_device_dict = single_device_model.state_dict()
    single_side_weight = torch.zeros((1008, 784))
    single_side_bias = torch.zeros(1008)
    for i in range(1008):
        single_side_bias[i] = device_side_bias[i]
        for j in range(784):
            single_side_weight[i, j] = device_side_weights[i, j]
    new_single_side_dict = {'fc1.weight': single_side_weight.to(device), 'fc1.bias': single_side_bias.to(device)}
    single_device_dict.update(new_single_side_dict)
    single_device_model.load_state_dict(single_device_dict)

    test_inference(server_model, cut_layer, single_device_model, device, test_loader)

    split_inference(server_model, cut_layer, device_models, device, test_loader)

    # aricomp-based split inference
    # noisy cut layer parameters set
    noisy_cut_layer = NoisyCutLayer().to(device)
    noisy_cut_layer_dict = noisy_cut_layer.state_dict()
    new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
    noisy_cut_layer_dict.update(new_cut_layer_dict)
    noisy_cut_layer.load_state_dict(noisy_cut_layer_dict)

    print('with aircomp')
    split_inference(server_model, noisy_cut_layer, device_models, device, test_loader)

    # print(server_model.state_dict())
    # print(device_models[0].parameters())
    # print(cut_layer.parameters())
    # print(entire_model_dict)
