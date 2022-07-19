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


class CustomDeviceNet(nn.Module):
    def __init__(self, next, pre):
        super(CustomDeviceNet, self).__init__()
        self.pre = pre
        self.next = next
        self.fc1 = nn.Linear(pre, next)

    def forward(self, x):
        x = x.view(-1, self.pre)
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
    def __init__(self, tau2=0., device=None):
        super(CutLayer, self).__init__()
        self.tau2 = tau2
        self.device = device
        self.fc2 = nn.Linear(1008, 500)

    def forward(self, x):
        x = x.view(-1, 1008)
        x = self.fc2(x)
        if self.tau2 != 0:
            noise = torch.normal(0, self.tau2, x.shape).to(self.device)
            x = x + noise
        x = F.relu(x)
        return x


# class NoisyCutLayer(nn.Module):
#     def __init__(self):
#         super(NoisyCutLayer, self).__init__()
#         self.tau2 = 0.1
#         self.fc2 = nn.Linear(1008, 500)
#
#     def forward(self, x):
#         x = x.view(-1, 1008)
#         noise = torch.normal(0, self.tau2, (x.shape[0], 500)).to(device)
#         x = F.relu(self.fc2(x) + noise)
#         return x


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
    return correct / len(test_loader.dataset)


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
    return correct / len(test_loader.dataset)


def split_inference(server_model, cut_layer, device_models, input_dim, pre_neurons_vector, device, test_loader):
    correct = 0
    n_devices = len(device_models)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            data = data.view(-1, input_dim)
            # data_list = torch.chunk(data, chunks=n_devices, dim=1)
            data_list = torch.split(data, pre_neurons_vector.tolist(), dim=1)
            device_side_output = None
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
    return correct / len(test_loader.dataset)


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


def plot_results(res, num_devices_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(num_devices_list, numpy.median(res[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xlabel('Number of Devices', fontsize=20)
    plt.ylabel('Inference Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/split_inference_neural_network_' + data_name + '_design.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


def different_inference_schemes(args):
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
    # num_devices_list = [3, 4, 5, 6, 7, 8, 9, 10]
    # num_devices_list = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    # num_devices_list = [2, 4, 7, 8, 14, 16, 28]
    num_devices_list = [2, 4, 7, 9, 12, 14, 17, 19]
    # num_devices_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    tau2_list = [0.2, 0.5, 1.0]
    repeat = 5
    data_name = 'FashionMNIST'
    legends = ['Scheme 1', 'Scheme 2', r"Scheme 3 $\sigma^{2}=0.1$", r"Scheme 3 $\sigma^{2}=0.5$",
               r"Scheme 3 $\sigma^{2}=1$"]
    results = numpy.zeros((len(tau2_list) + 2, repeat, len(num_devices_list)))
    for l in range(len(num_devices_list)):
        # normal model
        normal_model = Net().to(device)
        normal_model.load_state_dict(entire_model_dict)

        # split layers
        next_layer_neurons = split_layer(args.fc1_num_neurons, num_devices_list[l])
        pre_layer_neurons = split_layer(args.input_num_neurons, num_devices_list[l])

        # device-side models set
        device_models = []
        device_side_weights = entire_model_dict['fc1.weight']
        device_side_bias = entire_model_dict['fc1.bias']
        for i in range(num_devices_list[l]):
            device_model = CustomDeviceNet(next_layer_neurons[i], pre_layer_neurons[i]).to(device)
            device_models.append(device_model)

            device_dict = device_models[i].state_dict()
            fc1_weight = torch.zeros((next_layer_neurons[i], pre_layer_neurons[i]))
            fc1_bias = torch.zeros(next_layer_neurons[i])
            # print(get_num_neurons(pre_layer_neurons, i))
            for j in range(next_layer_neurons[i]):
                fc1_bias[j] = device_side_bias[get_num_neurons(next_layer_neurons, i) + j]
                for k in range(pre_layer_neurons[i]):
                    fc1_weight[j, k] = device_side_weights[
                        get_num_neurons(next_layer_neurons, i) + j, get_num_neurons(pre_layer_neurons, i) + k]

            new_device_dict = {'fc1.weight': fc1_weight.to(device), 'fc1.bias': fc1_bias.to(device)}
            device_dict.update(new_device_dict)
            device_models[i].load_state_dict(device_dict)

        # server-side model set
        server_model = ServerNet().to(device)
        server_dict = server_model.state_dict()
        new_server_dict = {k: v for k, v in entire_model_dict.items() if k in server_dict.keys()}
        server_dict.update(new_server_dict)
        server_model.load_state_dict(server_dict)

        # cut layer set
        pure_cut_layer = CutLayer(tau2=0.0, device=device).to(device)
        cut_layer_dict = pure_cut_layer.state_dict()
        new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
        cut_layer_dict.update(new_cut_layer_dict)
        pure_cut_layer.load_state_dict(cut_layer_dict)

        noisy_cut_layers = []
        for i in range(len(tau2_list)):
            cut_layer = CutLayer(tau2=tau2_list[i], device=device).to(device)
            cut_layer_dict = cut_layer.state_dict()
            new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
            cut_layer_dict.update(new_cut_layer_dict)
            cut_layer.load_state_dict(cut_layer_dict)
            noisy_cut_layers.append(cut_layer)



        for r in range(repeat):
            print('number of devices: ' + str(num_devices_list[l]) + ' repeat: ' + str(r))
            # normal inference
            results[0, r, l] = normal_inference(normal_model, device, test_loader)

            # split inference
            results[1, r, l] = split_inference(server_model, pure_cut_layer, device_models,
                                               args.input_num_neurons, pre_layer_neurons, device,
                                               test_loader)

            # aircomp-based split inference
            for i in range(len(tau2_list)):
                results[2 + i, r, l] = split_inference(server_model, noisy_cut_layers[i], device_models,
                                                       args.input_num_neurons, pre_layer_neurons, device,
                                                       test_loader)
    out_file_name = home_dir + 'Outputs/split_inference_neural_network_' + data_name + '_repeat_' + str(
        repeat) + '_design_results.npz'
    numpy.savez(out_file_name, res=results)
    plot_results(results, num_devices_list, data_name, legends)


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
    parser.add_argument('--input_num_neurons', type=int, default=784, metavar='N',
                        help='number of neurons in the input layer')
    parser.add_argument('--fc1_num_neurons', type=int, default=1008, metavar='N',
                        help='number of neurons in the fc1 layer')
    args = parser.parse_args()

    different_inference_schemes(args)

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    # device = torch.device("cuda" if use_cuda else "cpu")
    # #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # test_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=False,
    #                                                transform=transforms.Compose(
    #                                                    [transforms.ToTensor(), transforms.Normalize((0.1307,),
    #                                                                                                 (0.3081,))])),
    #                          batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #
    # entire_model_dict = torch.load('fashionmnist_fc.pt')
    # # normal inference
    # model = Net().to(device)
    # model.load_state_dict(entire_model_dict)
    # normal_inference(model, device, test_loader)
    #
    # # split inference
    # server_model = ServerNet().to(device)
    # n_devices = 2
    # device_models = []
    # for i in range(n_devices):
    #     device_model = DeviceNet().to(device)
    #     device_models.append(device_model)
    # tau2 = 0
    # cut_layer = CutLayer(tau2=tau2).to(device)
    #
    # # server-side model parameters set
    # server_dict = server_model.state_dict()
    # new_server_dict = {k: v for k, v in entire_model_dict.items() if k in server_dict.keys()}
    # server_dict.update(new_server_dict)
    # server_model.load_state_dict(server_dict)
    #
    # # cut layer parameters set
    # cut_layer_dict = cut_layer.state_dict()
    # new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
    # cut_layer_dict.update(new_cut_layer_dict)
    # cut_layer.load_state_dict(cut_layer_dict)
    #
    # # device-side model parameters set
    # device_side_weights = entire_model_dict['fc1.weight']
    # device_side_bias = entire_model_dict['fc1.bias']
    # for i in range(n_devices):
    #     device_dict = device_models[i].state_dict()
    #     fc1_weight = torch.zeros((504, 392))
    #     fc1_bias = torch.zeros(504)
    #     for j in range(504):
    #         fc1_bias[j] = device_side_bias[i * n_devices + j]
    #         for k in range(392):
    #             fc1_weight[j, k] = device_side_weights[i * n_devices + j, i * n_devices + k]
    #
    #     new_device_dict = {'fc1.weight': fc1_weight.to(device), 'fc1.bias': fc1_bias.to(device)}
    #     device_dict.update(new_device_dict)
    #     device_models[i].load_state_dict(device_dict)
    #
    # # test inference
    # single_device_model = SingleDeviceNet().to(device)
    # single_device_dict = single_device_model.state_dict()
    # single_side_weight = torch.zeros((1008, 784))
    # single_side_bias = torch.zeros(1008)
    # for i in range(1008):
    #     single_side_bias[i] = device_side_bias[i]
    #     for j in range(784):
    #         single_side_weight[i, j] = device_side_weights[i, j]
    # new_single_side_dict = {'fc1.weight': single_side_weight.to(device), 'fc1.bias': single_side_bias.to(device)}
    # single_device_dict.update(new_single_side_dict)
    # single_device_model.load_state_dict(single_device_dict)
    #
    # test_inference(server_model, cut_layer, single_device_model, device, test_loader)
    #
    # split_inference(server_model, cut_layer, device_models, device, test_loader)
    #
    # # aricomp-based split inference
    # # noisy cut layer parameters set
    # noisy_cut_layer = NoisyCutLayer().to(device)
    # noisy_cut_layer_dict = noisy_cut_layer.state_dict()
    # new_cut_layer_dict = {k: v for k, v in entire_model_dict.items() if k in cut_layer_dict.keys()}
    # noisy_cut_layer_dict.update(new_cut_layer_dict)
    # noisy_cut_layer.load_state_dict(noisy_cut_layer_dict)
    #
    # print('with aircomp')
    # split_inference(server_model, noisy_cut_layer, device_models, device, test_loader)

    # print(server_model.state_dict())
    # print(device_models[0].parameters())
    # print(cut_layer.parameters())
    # print(entire_model_dict)
    # next = 1000
    # pre = 784
    # testNet = CutLayer(tau2=0.1, device=device)
    # x = torch.zeros((1, 1008))
    # print(testNet.forward(x))
