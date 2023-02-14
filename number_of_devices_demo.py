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
    random_system_param_v2
from split_inference import WirelessSplitNet, train, test, split_layer, get_num_neurons, FashionMNIST_training

home_dir = './'
sys.path.append(home_dir)


class SplitNetVersusDevices(WirelessSplitNet):
    def __init__(self, next_layer_neurons, pre_layer_neurons, device=None):
        super(SplitNetVersusDevices, self).__init__(next_layer_neurons, pre_layer_neurons, device=device)
        self.active_devices = None

    def set_active_devices(self, active_devices=None):
        self.active_devices = active_devices

    def forward(self, x):
        x = x.view(-1, 784)
        x_list = torch.split(x, self.pre_layer_neurons.tolist(), dim=1)
        # fc1_output = None

        # device-side forward
        device_side_output_list = list()
        for i in range(self.n_devices):
            if i in self.active_devices:
                x_data = x_list[i]
                sub_model_output = self.sub_models[i](x_data)

                sub_model_output = self.layer_norms[i](sub_model_output)
                # print(torch.mean(sub_model_output))
                # print(torch.linalg.norm(sub_model_output))

                # if i == 0:
                #     fc1_output = sub_model_output
                # else:
                #     fc1_output = torch.cat([fc1_output, sub_model_output], dim=1)
                device_side_output = self.cut_layer[i](sub_model_output)
                device_side_output_list.append(device_side_output)
            else:
                device_side_output_list.append(torch.zeros((1000, 32)).to(self.device))

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


def plot_results(res, number_of_devices_list, tau2, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(number_of_devices_list, numpy.median(res[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xticks(number_of_devices_list)
    plt.xlabel('Number of Devices', fontsize=20)
    plt.ylabel('Inference Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(tau2) + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


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
    # unbalanced
    next_layer_neurons_list = [21, 28, 35, 21, 28, 35, 21, 28, 35, 21, 28, 35, 28, 28]
    pre_layer_neurons_list = [42, 56, 70, 42, 56, 70, 42, 56, 70, 42, 56, 70, 56, 56]
    next_layer_neurons = numpy.array(next_layer_neurons_list)
    pre_layer_neurons = numpy.array(pre_layer_neurons_list)

    # load model
    model_state_dict = torch.load('multi_modality_fashionmnist_n_devices_' + str(n_devices) + '.pt')
    model = SplitNetVersusDevices(next_layer_neurons, pre_layer_neurons, device=device).to(device)
    wireless_split_net_dict = model.state_dict()
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
    tau2_list = [1, 2, 4]
    eta_list = [5e-5, 4e-5, 2e-5]
    number_of_devices_list = [2, 4, 6, 8, 10, 12, 14]
    w_mat = numpy.zeros((n_devices, J))
    ini_h_mat = abs(numpy.random.randn(n_devices, K, m))
    for n in range(n_devices):
        for j in range(J):
            tmp = 0
            for i in range(next_layer_neurons[n]):
                tmp += fc2_weights_list[n][j, i] ** 2
            w_mat[n, j] = tmp
    repeat = 25
    data_name = 'fashionMNIST'
    # data_name = 'cifar10'
    legends = ['Scheme 1', 'Scheme 2']
    results = numpy.zeros((2, repeat, len(number_of_devices_list)))
    objectives = numpy.zeros((2, repeat, len(number_of_devices_list)))
    stored_results = numpy.zeros((2, len(number_of_devices_list)))
    stored_objectives = numpy.zeros((2, len(number_of_devices_list)))

    for iter_tau2 in range(len(tau2_list)):
        for r in range(repeat):
            h_mat = ini_h_mat.copy()
            for n in range(n_devices):
                # subcarrier_scale_list = 4.5 * numpy.random.random_sample(K) + 0.5
                subcarrier_scale_list = numpy.zeros(K)
                subcarrier_scale_list[0:int(K / 4)] = 0.1 * numpy.random.random_sample(int(K / 4)) + 0.1
                # subcarrier_scale_list[int(K / 8):] = 10 * numpy.random.random_sample(K - int(K / 8)) + 10
                subcarrier_scale_list[int(K / 4):] = 1
                subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
                tmp_subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
                # PL = (10 ** 7) * ((distance_list[n] / 1) ** (-3.76))
                # h_mat[n] = PL * h_mat[n]
                for k in range(K):
                    h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]
                # print(h_mat[n])

            for i in range(len(number_of_devices_list)):
                print('---number of devices: ' + str(number_of_devices_list[i]))

                active_devices = []
                random_permutation = numpy.random.permutation(n_devices)
                for perm_iter in range(number_of_devices_list[i]):
                    active_devices.append(random_permutation[perm_iter])

                objectives[0, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[iter_tau2], P, OPTIMIZED, eta=eta_list[iter_tau2])
                # objectives[0, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[iter_tau2], P, PURE)
                model.set_active_devices(active_devices)
                stored_objectives[0, i] = objectives[0, r, i]
                results[0, r, i] = test(model, device, test_loader)
                stored_results[0, i] = results[0, r, i]
                objectives[1, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[iter_tau2], P, PURE)
                model.set_active_devices(range(n_devices))
                stored_objectives[1, i] = objectives[1, r, i]
                results[1, r, i] = test(model, device, test_loader)
                stored_results[1, i] = results[1, r, i]
                # objectives[2, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[i], P, PURE)
                # stored_objectives[2, i] = objectives[2, r, i]
                # pure_model = MultiModalityNet(next_layer_neurons, pre_layer_neurons, tau2=0., device=device,
                #                               dataset='FashionMNIST').to(device)
                # pure_model.load_state_dict(model_state_dict)
                # results[2, r, i] = test(model, device, test_loader)
                # stored_results[2, i] = results[2, r, i]

            out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(
                tau2_list[iter_tau2]) + '_repeat_' + str(r+50) + '_partial_results.npz'
            numpy.savez(out_file_name, res=stored_results, obj=stored_objectives)
        out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(
            tau2_list[iter_tau2]) + '_repeat_' + str(repeat) + '_total_results.npz'
        numpy.savez(out_file_name, res=results, obj=objectives)
        plot_results(results, number_of_devices_list, tau2_list[iter_tau2], data_name, legends)
