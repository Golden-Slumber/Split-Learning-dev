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


class ServerNet(nn.Module):
    def __init__(self, dataset=None):
        super(ServerNet, self).__init__()
        self.dataset = dataset
        if dataset == 'FashionMNIST':
            self.fc3 = nn.Linear(196, 98, bias=True)
            self.fc4 = nn.Linear(98, 49, bias=True)
            self.output = nn.Linear(49, 10, bias=True)
        elif dataset == 'cifar10':
            self.fc3 = nn.Linear(768, 192, bias=True)
            self.fc4 = nn.Linear(192, 48, bias=True)
            self.output = nn.Linear(48, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.dataset == 'FashionMNIST':
            x = x.view(-1, 196)
        elif self.dataset == 'cifar10':
            x = x.view(-1, 768)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.softmax(self.output(x))


class CustomDeviceNet(nn.Module):
    def __init__(self, num_next, num_pre):
        super(CustomDeviceNet, self).__init__()
        self.num_pre = num_pre
        self.num_next = num_next
        self.fc1 = nn.Linear(num_pre, num_next)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.num_pre)
        x = self.relu(self.fc1(x))
        return x


class CutLayer(nn.Module):
    def __init__(self, tau2=0.0, device=None, dataset=None):
        super(CutLayer, self).__init__()
        self.tau2 = tau2
        self.device = device
        self.dataset = dataset
        if dataset == 'FashionMNIST':
            self.fc2 = nn.Linear(392, 196)
        elif dataset == 'cifar10':
            self.fc2 = nn.Linear(1536, 768)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.dataset == 'FashionMNIST':
            x = x.view(-1, 392)
        elif self.dataset == 'cifar10':
            x = x.view(-1, 1536)
        x = self.fc2(x)
        if self.tau2 != 0.0:
            noise = torch.normal(0, self.tau2, x.shape).to(self.device)
            x = x + noise
            # print(noise)
        x = self.relu(x)
        return x


class MultiModalityNet(nn.Module):
    def __init__(self, next_layer_neurons, pre_layer_neurons, tau2=0.0, device=None, dataset=None):
        super(MultiModalityNet, self).__init__()
        self.tau2 = tau2
        self.device = device
        self.dataset = dataset
        self.next_layer_neurons = next_layer_neurons
        self.pre_layer_neurons = pre_layer_neurons
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.layer_norms = [nn.LayerNorm(next_layer_neurons[j]).to(device) for j in range(len(next_layer_neurons))]

        # self.sub_models = []
        # for i in range(len(next_layer_neurons)):
        #     sub_model = nn.Linear(pre_layer_neurons[i], next_layer_neurons[i]).to(device)
        #     self.sub_models.append(sub_model)
        self.sub_models = nn.ModuleList(
            [nn.Sequential(nn.Linear(pre_layer_neurons[j], next_layer_neurons[j]), nn.ReLU()) for j in
             range(len(next_layer_neurons))])

        if dataset == 'FashionMNIST':
            self.fc2 = nn.Linear(392, 196)
            self.fc3 = nn.Linear(196, 98, bias=True)
            self.fc4 = nn.Linear(98, 49, bias=True)
            self.output = nn.Linear(49, 10, bias=True)
        elif dataset == 'cifar10':
            self.fc2 = nn.Linear(1536, 768)
            self.fc3 = nn.Linear(768, 192, bias=True)
            self.fc4 = nn.Linear(192, 48, bias=True)
            self.output = nn.Linear(48, 10, bias=True)

    def forward(self, x):
        if self.dataset == 'FashionMNIST':
            x = x.view(-1, 784)
        elif self.dataset == 'cifar10':
            x = x.view(-1, 3072)
        x_list = torch.split(x, self.pre_layer_neurons.tolist(), dim=1)
        fc1_output = None
        for i in range(len(self.next_layer_neurons)):
            x_data = x_list[i]
            sub_model_output = self.sub_models[i](x_data)

            sub_model_output = self.layer_norms[i](sub_model_output)
            # print(torch.mean(sub_model_output))
            # print(torch.linalg.norm(sub_model_output))

            if i == 0:
                fc1_output = sub_model_output
            else:
                fc1_output = torch.cat([fc1_output, sub_model_output], dim=1)

        x = self.fc2(fc1_output)
        if self.tau2 != 0.0:
            noise = torch.normal(0, self.tau2, x.shape).to(self.device)
            x = x + noise
            print(torch.linalg.norm(x))
            print(torch.linalg.norm(noise))
        x = self.relu(x)

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.softmax(self.output(x))


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


def test(model, device, test_loader):
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
    return correct / len(test_loader.dataset)


def split_layer(num_tot_neurons, num_devices, balanced=True):
    if balanced:
        neurons_vector = numpy.ones(num_devices).astype(int) * int(num_tot_neurons / num_devices)
    else:
        neurons_vector = numpy.random.random(num_devices)
        neurons_vector *= num_tot_neurons * 0.3 / neurons_vector.sum()
        neurons_vector = neurons_vector.astype(int) + int(num_tot_neurons / num_devices * 0.7)

    extra = num_tot_neurons - neurons_vector.sum()
    i = 0
    while extra > 0:
        neurons_vector[i] += 1
        extra -= 1
        i += 1
        i %= num_devices

    print(neurons_vector)
    return neurons_vector


def get_num_neurons(neurons_vectors, idx):
    cnt = 0
    for i in range(idx):
        cnt += neurons_vectors[i]
    return cnt


def plot_results(res, tau2_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(tau2_list, numpy.median(res[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xticks(tau2_list)
    plt.xlabel(r"$\sigma^{2}$", fontsize=20)
    plt.ylabel('Inference Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/multi_modality_neural_network_' + data_name + '_normalized.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


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

    n_devices = 4
    next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)
    model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device, dataset='FashionMNIST').to(
        device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'multi_modality_fashionmnist_normalized_fc.pt')


def cifar10_training(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                  (0.5, 0.5, 0.5))])),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                  (0.5, 0.5, 0.5))])),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    n_devices = 4
    next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)
    model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device, dataset='cifar10').to(
        device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'multi_modality_cifar10_normalized_fc.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion MNIST')
    # parser = argparse.ArgumentParser(description='cifar10')
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
    parser.add_argument('--input_num_neurons', type=int, default=784, metavar='N',
                        help='number of neurons in the input layer')
    parser.add_argument('--fc1_num_neurons', type=int, default=392, metavar='N',
                        help='number of neurons in the fc1 layer')
    # parser.add_argument('--input_num_neurons', type=int, default=3072, metavar='N',
    #                     help='number of neurons in the input layer')
    # parser.add_argument('--fc1_num_neurons', type=int, default=1536, metavar='N',
    #                     help='number of neurons in the fc1 layer')
    args = parser.parse_args()

    # FashionMNIST_training(args)
    # cifar10_training(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.ToTensor(), transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))])),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(datasets.CIFAR10(root='./Resources/', train=False,
    #                                                transform=transforms.Compose(
    #                                                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                                               (0.5, 0.5, 0.5))])),
    #                          batch_size=args.test_batch_size, shuffle=True, **kwargs)

    n_devices = 4
    next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)

    # tau2_list = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0]
    tau2_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]

    model_state_dict = torch.load('multi_modality_fashionmnist_normalized_fc.pt')
    print(model_state_dict)
    model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device,
                             dataset='FashionMNIST').to(device)

    # model_state_dict = torch.load('multi_modality_cifar10_normalized_fc.pt')
    # model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device,
    #                          dataset='cifar10').to(device)

    model.load_state_dict(model_state_dict)

    repeat = 5
    data_name = 'fashionMNIST'
    # data_name = 'cifar10'
    legends = ['Scheme 1', 'Scheme 2']
    results = numpy.zeros((2, repeat, len(tau2_list)))

    for i in range(len(tau2_list)):
        noisy_model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=tau2_list[i], device=device,
                                       dataset='FashionMNIST').to(device)
        # noisy_model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=tau2_list[i], device=device,
        #                                dataset='cifar10').to(device)
        noisy_model.load_state_dict(model_state_dict)
        for r in range(repeat):
            results[0, r, i] = test(model, device, test_loader)
            results[1, r, i] = test(noisy_model, device, test_loader)
    out_file_name = home_dir + 'Outputs/multi_modality_neural_network_' + data_name + '_repeat_' + str(
        repeat) + '_normalized_results.npz'
    numpy.savez(out_file_name, res=results)
    # npz_file = numpy.load(out_file_name, allow_pickle=True)
    # results = npz_file['res']
    plot_results(results, tau2_list, data_name, legends)
