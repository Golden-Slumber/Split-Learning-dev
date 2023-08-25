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
from multi_modality_nural_network import MultiModalityNet
from system_optimization import alternating_optimization_v2, random_system_param, alternating_optimization_v3
from revised_system_optimization import alternating_optimization_framework, fixed_subcarrier_allocation, \
    random_system_param_v2, BnB_alternating_optimization_framework
from Hungarian_based_system_optimization import graph_based_alternating_optimization_framework, \
    subcarrier_aware_optimization, power_aware_optimization

home_dir = './'
sys.path.append(home_dir)


class WirelessSplitNet(nn.Module):
    def __init__(self, next_layer_neurons, pre_layer_neurons, device=None):
        super(WirelessSplitNet, self).__init__()
        self.device = device
        self.next_layer_neurons = next_layer_neurons
        self.pre_layer_neurons = pre_layer_neurons
        self.n_devices = len(self.next_layer_neurons)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        # self.layer_norms = [nn.LayerNorm(next_layer_neurons[j]).to(device) for j in range(len(next_layer_neurons))]
        self.w_mat = None
        self.h_mat = None
        self.tau2 = None
        self.P = None
        self.indicator_mat = None
        self.b_mat = None
        self.a_list = list()

        # device-side models
        self.sub_models = nn.ModuleList(
            [nn.Sequential(nn.Linear(pre_layer_neurons[j], next_layer_neurons[j]), nn.ReLU()) for j in
             range(len(next_layer_neurons))])

        # cut layer
        self.cut_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(next_layer_neurons[j], 32)) for j in
             range(len(next_layer_neurons))])
        self.cut_layer_bias = None

        # server-side model
        self.fc3 = nn.Linear(32, 98, bias=True)
        self.fc4 = nn.Linear(98, 49, bias=True)
        self.output = nn.Linear(49, 10, bias=True)

    def set_system_params(self, w_mat, h_mat, tau2, P, mode, eta=None):
        self.w_mat = w_mat
        self.h_mat = h_mat
        self.tau2 = tau2
        self.P = P
        self.mode = mode
        mse = 0
        if self.mode != PURE:
            if self.mode == OPTIMIZED:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = alternating_optimization_framework(self.w_mat,
                                                                                                   self.h_mat,
                                                                                                   self.tau2, self.P,
                                                                                                   eta=eta)
            elif self.mode == RANDOM:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = fixed_subcarrier_allocation(self.w_mat, self.h_mat,
                                                                                            self.tau2, self.P)
            elif self.mode == RANDOM2:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = random_system_param_v2(self.w_mat, self.h_mat,
                                                                                       self.tau2, self.P)
            elif self.mode == BNB:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = BnB_alternating_optimization_framework(self.w_mat,
                                                                                                       self.h_mat,
                                                                                                       self.tau2,
                                                                                                       self.P)
            elif self.mode == SUBCARRIER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = subcarrier_aware_optimization(self.w_mat, self.h_mat,
                                                                                              self.tau2, self.P)
            elif self.mode == POWER_AWARE:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = power_aware_optimization(self.w_mat, self.h_mat,
                                                                                         self.tau2, self.P)
            elif self.mode == GRAPH:
                tmp_indicator_mat, tmp_b_mat, tmp_a_list, mse = graph_based_alternating_optimization_framework(
                    self.w_mat, self.h_mat,
                    self.tau2, self.P, max_iter=20)
                # print(tmp_indicator_mat)
            # print(tmp_b_mat)
            # print(tmp_a_list)
            self.indicator_mat = None
            self.b_mat = None
            self.a_list = list()

            self.indicator_mat = torch.from_numpy(tmp_indicator_mat).to(self.device)
            self.b_mat = torch.from_numpy(tmp_b_mat).to(self.device)
            for j in range(32):
                self.a_list.append(torch.from_numpy(tmp_a_list[j]).to(self.device))

        return mse

    def forward(self, x):
        x = x.view(-1, 784)
        x_list = torch.split(x, self.pre_layer_neurons.tolist(), dim=1)
        # fc1_output = None

        # device-side forward
        device_side_output_list = list()
        sub_model_output_list = list()
        mean = 0
        var = 0
        for i in range(len(self.next_layer_neurons)):
            x_data = x_list[i]
            sub_model_output = self.sub_models[i](x_data)

            cur_mean = sub_model_output.mean()
            var += (sub_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()

            sub_model_output_list.append(sub_model_output)

            # sub_model_output = self.layer_norms[i](sub_model_output)
            # print(torch.mean(sub_model_output))
            # print(torch.linalg.norm(sub_model_output))

            # if i == 0:
            #     fc1_output = sub_model_output
            # else:
            #     fc1_output = torch.cat([fc1_output, sub_model_output], dim=1)

        mean /= len(self.next_layer_neurons)
        var /= len(self.next_layer_neurons)

        for i in range(len(self.next_layer_neurons)):
            sub_model_output = (sub_model_output_list[i] - mean) / numpy.sqrt(var + 1e-6)
            device_side_output = self.cut_layer[i](sub_model_output)
            # print(device_side_output.mean())
            device_side_output_list.append(device_side_output)

        # over-the-air aggregation
        if self.mode != PURE:
            tmp_h_mat = torch.from_numpy(self.h_mat).to(self.device)
            received_signal = torch.zeros((1000, 32)).to(self.device)
            for n in range(self.n_devices):
                transmit_signal = torch.multiply(device_side_output_list[n], self.b_mat[n])
                h_vec = torch.zeros(32).to(self.device)
                for j in range(32):
                    for k in range(self.indicator_mat.shape[1]):
                        if self.indicator_mat[j, k] == 1:
                            h_vec[j] = torch.sum(torch.mm(self.a_list[j].T, tmp_h_mat[n, k].reshape((5, 1))))
                received_signal += torch.multiply(transmit_signal, h_vec)
            noise = torch.normal(0, self.tau2, (32, 5)).to(self.device)
            scaled_noise = torch.zeros(received_signal.shape).to(self.device)
            # for j in range(32):
            #     scaled_noise[:, j] = torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
            # received_signal = received_signal + scaled_noise
            for j in range(32):
                received_signal[:, j] += torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
            x = received_signal + self.cut_layer_bias
            x = self.relu(x)
        else:
            # print(device_side_output_list)
            x = sum(device_side_output_list) + self.cut_layer_bias
            x = self.relu(x)

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.softmax(self.output(x))

    def get_statistics(self, x, n_devices, J):
        variance_mat = numpy.zeros((n_devices, J))
        x = x.view(-1, 784)
        x_list = torch.split(x, self.pre_layer_neurons.tolist(), dim=1)
        # fc1_output = None

        # device-side forward
        device_side_output_list = list()
        sub_model_output_list = list()
        mean = 0
        var = 0
        for i in range(len(self.next_layer_neurons)):
            x_data = x_list[i]
            sub_model_output = self.sub_models[i](x_data)

            # print(sub_model_output)
            cur_mean = sub_model_output.mean()
            var += (sub_model_output - cur_mean).pow(2).mean().item()
            mean += cur_mean.item()

            sub_model_output_list.append(sub_model_output)

            # sub_model_output = self.layer_norms[i](sub_model_output)
            # print(torch.mean(sub_model_output))
            # print(torch.linalg.norm(sub_model_output))

            # if i == 0:
            #     fc1_output = sub_model_output
            # else:
            #     fc1_output = torch.cat([fc1_output, sub_model_output], dim=1)

        mean /= len(self.next_layer_neurons)
        var /= len(self.next_layer_neurons)

        for i in range(len(self.next_layer_neurons)):
            sub_model_output = (sub_model_output_list[i] - mean) / numpy.sqrt(var + 1e-6)
            device_side_output = self.cut_layer[i](sub_model_output)
            device_side_output_list.append(device_side_output)

            variance_mat[i, :] = torch.var(device_side_output, 0, correction=0).cpu().numpy()

        return variance_mat


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


def plot_results(res, obj, tau2_list, data_name, legends):
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

    image_name = home_dir + 'Outputs/MLP_demo_' + data_name + '_accuracy_1.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(tau2_list, numpy.median(obj[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xticks(tau2_list)
    plt.xlabel(r"$\sigma^{2}$", fontsize=20)
    plt.ylabel('Objective Value', fontsize=20)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/MLP_demo_' + data_name + '_objective_1.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


def FashionMNIST_training(args, n_devices, next_layer_neurons, pre_layer_neurons):
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

    # n_devices = 4
    # next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=True)
    # pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=True)
    model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device, dataset='FashionMNIST').to(
        device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), 'vertically_split_fashionmnist_n_devices_' + str(n_devices) + '.pt')


def tau_mat_processing(args, model, n_devices, J):
    tau_mat = numpy.zeros((n_devices, J))

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

    n_batches = 0
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            partial_tau_mat = model.get_statistics(data, n_devices, J)
            tau_mat += partial_tau_mat
            n_batches += 1
    return tau_mat / n_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion MNIST')
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

    n_devices = 14
    # next_layer_neurons = split_layer(args.fc1_num_neurons, n_devices, balanced=False)
    # pre_layer_neurons = split_layer(args.input_num_neurons, n_devices, balanced=False)
    # unbalanced
    next_layer_neurons_list = [21, 28, 35, 21, 28, 35, 21, 28, 35, 21, 28, 35, 28, 28]
    pre_layer_neurons_list = [42, 56, 70, 42, 56, 70, 42, 56, 70, 42, 56, 70, 56, 56]
    next_layer_neurons = numpy.array(next_layer_neurons_list)
    pre_layer_neurons = numpy.array(pre_layer_neurons_list)

    train_flag = True
    if train_flag:
        train_loader = DataLoader(datasets.FashionMNIST(root='./Resources/', train=True, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                                      transforms.Normalize((0.1307,),
                                                                                                           (
                                                                                                           0.3081,))])),
                                  batch_size=args.batch_size, shuffle=True, **kwargs)
        image, label = next(iter(train_loader))
        plt.imshow(image[1][0], cmap="gray")
        plt.show()
        print(label[1])
        # FashionMNIST_training(args, n_devices, next_layer_neurons, pre_layer_neurons)
    else:
        # load model
        model_state_dict = torch.load('vertically_split_fashionmnist_n_devices_' + str(n_devices) + '.pt')
        model = WirelessSplitNet(next_layer_neurons, pre_layer_neurons, device=device).to(device)
        wireless_split_net_dict = model.state_dict()
        # print(model_state_dict.keys())
        # print(wireless_split_net_dict.keys())
        # print(wireless_split_net_dict['cut_layer.0.0.weight'].shape)
        new_dict = {k: v for k, v in model_state_dict.items() if k in wireless_split_net_dict.keys()}
        fc2_weights = model_state_dict['fc2.weight']
        fc2_bias = model_state_dict['fc2.bias']
        model.cut_layer_bias = fc2_bias.to(device)
        fc2_weights_list = torch.split(fc2_weights, next_layer_neurons.tolist(), dim=1)
        # print(fc2_weights_list)
        for i in range(len(next_layer_neurons)):
            key = 'cut_layer.' + str(i) + '.0.weight'
            new_dict[key] = fc2_weights_list[i]
            key = 'cut_layer.' + str(i) + '.0.bias'
            new_dict[key] = torch.zeros(32)
        wireless_split_net_dict.update(new_dict)
        model.load_state_dict(wireless_split_net_dict)

        # system parameters
        J = 32
        K = 32
        m = 5
        P = 10
        # tau2_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        # tau2_list = [0.02, 0.04, 0.06, 0.08, 0.1]
        # tau2_list = [0.02, 0.22, 0.42, 0.62, 0.82]
        # tau2_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        # tau2_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        # tau2_list = [1, 2, 3, 4, 5, 6, 7]
        # tau2_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        tau2_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        # tau2_list = [0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3]
        # tau2_list = [0.82]
        # eta_list = [0.27, 0.2, 0.1, 0.05, 0.08]
        # eta_list = [0.08, 0.05, 0.02, 0.01, 0.008, 0.005, 0.002, 0.001, 0.0008, 0.0005]
        eta_list = [5e-5, 5e-5, 4e-5, 4e-5, 3e-5, 3e-5, 2e-5, 2e-5]
        # eta_list = [2e-5]

        # Estimation based on training samples
        # w_mat = numpy.zeros((n_devices, J))
        # distance_list = numpy.random.randint(1, 20, size=n_devices)
        ini_h_mat = abs(numpy.random.randn(n_devices, K, m))
        # for n in range(n_devices):
        #     PL = (10 ** 3) * ((distance_list[n] / 1) ** (-3.76))
        #     h_mat[n] = distance_list[n] * h_mat[n] / 10
        # for n in range(n_devices):
        #     for j in range(J):
        #         tmp = 0
        #         for i in range(next_layer_neurons[n]):
        #             tmp += fc2_weights_list[n][j, i] ** 2
        #         w_mat[n, j] = tmp
        w_mat = tau_mat_processing(args, model, n_devices, J)

        repeat = 100
        data_name = 'fashionMNIST'
        # data_name = 'cifar10'
        legends = ['Scheme 1', 'Scheme 2', 'Scheme 3']
        # legends = ['Scheme 1', 'Scheme 2']
        results = numpy.zeros((3, repeat, len(tau2_list)))
        objectives = numpy.zeros((3, repeat, len(tau2_list)))
        stored_results = numpy.zeros((3, len(tau2_list)))
        stored_objectives = numpy.zeros((3, len(tau2_list)))
        # stored_results = numpy.zeros((1, len(tau2_list)))
        # stored_objectives = numpy.zeros((1, len(tau2_list)))

        for r in range(repeat):
            # h_mat = abs(numpy.random.randn(n_devices, K, m))
            h_mat = ini_h_mat.copy()
            for n in range(n_devices):
                subcarrier_scale_list = numpy.zeros(K)
                subcarrier_scale_list[0:int(K / 4)] = 0.1 * numpy.random.random_sample(int(K / 4)) + 0.1
                # # subcarrier_scale_list[int(K / 4):] = 5 * numpy.random.random_sample(int(K / 2)) + 5
                subcarrier_scale_list[int(K / 4):int(K / 2)] = 0.2 * numpy.random.random_sample(int(K / 4)) + 0.2
                subcarrier_scale_list[int(K / 2):] = 1
                # subcarrier_scale_list[:] = numpy.random.random_sample(K) + 0.5
                subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
                # tmp_subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
                for k in range(K):
                    h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]

            for i in range(len(tau2_list)):
                print('iteration ' + str(r) + ' - noise variance: ' + str(tau2_list[i]))

                # objectives[0, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[i], P, OPTIMIZED, eta=eta_list[i])
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

                # pure_model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device,
                #                               dataset='FashionMNIST').to(device)
                # pure_model.load_state_dict(model_state_dict)
                # results[2, r, i] = test(model, device, test_loader)
                # stored_results[2, i] = results[2, r, i]
            out_file_name = home_dir + 'Outputs/MLP_demo_' + data_name + '_nvar-range_' + str(
                tau2_list[0]) + '-' + str(tau2_list[-1]) + '_repeat_' + str(r) + '_results.npz'
            numpy.savez(out_file_name, res=stored_results, obj=stored_objectives)
        # out_file_name = home_dir + 'Outputs/aircomp_based_split_inference_' + data_name + '_repeats_' + str(
        #     repeat) + '_total_results.npz'
        # numpy.savez(out_file_name, res=results, obj=objectives)
        plot_results(results, objectives, tau2_list, data_name, legends)
