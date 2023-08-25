import pickle
import sys
import numpy
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

from split_inference import WirelessSplitNet, tau_mat_processing
from CNN_demo import SplitCNN
from Resnet_demo import RevisedSplitResnet

home_dir = './'
sys.path.append(home_dir)


class PartialDevicesSplitMLP(WirelessSplitNet):
    def __init__(self, next_layer_neurons, pre_layer_neurons, device=None):
        super(PartialDevicesSplitMLP, self).__init__(next_layer_neurons, pre_layer_neurons, device=device)
        self.active_devices = None

    def set_active_devices(self, active_devices):
        self.active_devices = active_devices

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
            if i in self.active_devices:
                x_data = x_list[i]
                sub_model_output = self.sub_models[i](x_data)

                cur_mean = sub_model_output.mean()
                var += (sub_model_output - cur_mean).pow(2).mean().item()
                mean += cur_mean.item()

                sub_model_output_list.append(sub_model_output)
            else:
                sub_model_output_list.append(torch.zeros((1000, 32)).to(self.device))
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
            if i in self.active_devices:
                sub_model_output = (sub_model_output_list[i] - mean) / numpy.sqrt(var + 1e-6)
                device_side_output = self.cut_layer[i](sub_model_output)
                # print(device_side_output.mean())
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

class PartialDevicesSplitCNN(SplitCNN):
    def __init__(self, img_size_list, device=None):
        super(PartialDevicesSplitCNN, self).__init__(img_size_list, device)
        self.active_devices = None

    def set_active_devices(self, active_devices):
        self.active_devices = active_devices

    def forward(self, x):
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            if n in self.active_devices:
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
                cur_mean = device_model_output.mean()
                var += (device_model_output - cur_mean).pow(2).mean().item()
                mean += cur_mean.item()
                device_model_outputs.append(device_model_output)
            else:
                device_model_outputs.append(torch.zeros((x.size(0), self.img_size_list[n] ** 2)).to(self.device))
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            if n in self.active_devices:
                device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
                device_model_outputs[n] = self.cut_layer[n](device_model_outputs[n])
            else:
                device_model_outputs[n] = torch.zeros((x.size(0), 128)).to(self.device)

        if self.sigma2 == 0:
            server_side_input = sum(device_model_outputs)
            res = self.relu(server_side_input)
            res = self.relu(self.fc1(res))
            # res = self.relu(self.fc2(res))
            return self.softmax(self.output(res))
        else:
            tmp_h_mat = torch.from_numpy(self.h_mat).to(self.device)
            received_signal = torch.zeros((1800, 128)).to(self.device)
            for n in range(self.n_devices):
                transmit_signal = torch.multiply(device_model_outputs[n], self.b_mat[n])
                h_vec = torch.zeros(128).to(self.device)
                for j in range(128):
                    for k in range(self.indicator_mat.shape[1]):
                        if self.indicator_mat[j, k] == 1:
                            h_vec[j] = torch.sum(torch.mm(self.a_list[j].T, tmp_h_mat[n, k].reshape((5, 1))))
                received_signal += torch.multiply(transmit_signal, h_vec)
            noise = torch.normal(0, self.sigma2, (128, 5)).to(self.device)
            for j in range(128):
                received_signal[:, j] += torch.sum(torch.mm(self.a_list[j].T.float(), noise[j].reshape((5, 1))))
            res = self.relu(received_signal)
            res = self.relu(self.fc1(res))
            # res = self.relu(self.fc2(res))
            return self.softmax(self.output(res))

class PartialDevicesSplitResnet(RevisedSplitResnet):
    def __init__(self, img_size_list, device=None):
        super(PartialDevicesSplitResnet, self).__init__(img_size_list, device)
        self.active_devices = None

    def set_active_devices(self, active_devices):
        self.active_devices = active_devices

    def forward(self, x):
        device_model_outputs = list()
        mean = 0
        var = 0
        for n in range(self.n_devices):
            if n in self.active_devices:
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
            else:
                device_model_outputs.append(torch.zeros((x.size(0), 64)).to(self.device))
        mean /= self.n_devices
        var /= self.n_devices

        for n in range(self.n_devices):
            if n in self.active_devices:
                device_model_outputs[n] = (device_model_outputs[n] - mean) / numpy.sqrt(var + 1e-6)
                device_model_outputs[n] = self.cut_layers[n](device_model_outputs[n])
            else:
                device_model_outputs[n] = torch.zeros((x.size(0), 64)).to(self.device)

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

def FashionMNIST_tau_mat_processing(args, model, n_devices, J):
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

    n_batches = 0
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            partial_tau_mat = model.get_statistics(data, n_devices, J)
            tau_mat += partial_tau_mat
            n_batches += 1
    return tau_mat / n_batches

def FashionMNIST_test(model, device, test_loader):
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

def EuroSAT_tau_mat_processing(args, model, n_devices, J):
    tau_mat = numpy.zeros((n_devices, J))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    n_batches = 0
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            partial_tau_mat = model.get_statistics(data, J)
            tau_mat += partial_tau_mat
            n_batches += 1
    return tau_mat / n_batches

def EuroSAT_test(model, device, test_loader):
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

def plot_results(results, number_of_devices_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(number_of_devices_list, numpy.median(results[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(number_of_devices_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Number of Participating Devices', fontsize=25)
    plt.ylabel('Inference Accuracy', fontsize=25)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/PartialDevices_demo_' + data_name + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


def MLP_versus_devices_demo():
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
    model_state_dict = torch.load('vertically_split_fashionmnist_n_devices_' + str(n_devices) + '.pt')
    model = PartialDevicesSplitMLP(next_layer_neurons, pre_layer_neurons, device=device).to(device)
    wireless_split_net_dict = model.state_dict()
    new_dict = {k: v for k, v in model_state_dict.items() if k in wireless_split_net_dict.keys()}
    fc2_weights = model_state_dict['fc2.weight']
    fc2_bias = model_state_dict['fc2.bias']
    model.cut_layer_bias = fc2_bias.to(device)
    fc2_weights_list = torch.split(fc2_weights, next_layer_neurons.tolist(), dim=1)
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

    tau2_list = [0, 0.5, 1, 2]
    number_of_devices_list = [2, 4, 6, 8, 10, 12, 14]
    ini_h_mat = abs(numpy.random.randn(n_devices, K, m))
    w_mat = FashionMNIST_tau_mat_processing(args, model, n_devices, J)

    repeat = 50
    data_name = 'fashionMNIST'
    legends = [r"Proposed Approach: $\sigma^{2}=0$", r"Proposed Approach: $\sigma^{2}=0.5$",
               r"Proposed Approach: $\sigma^{2}=1$", r"Proposed Approach: $\sigma^{2}=2$"]
    results = numpy.zeros((4, repeat, len(number_of_devices_list)))
    # objectives = numpy.zeros((4, repeat, len(tau2_list)))
    stored_results = numpy.zeros((4, len(number_of_devices_list)))
    # stored_objectives = numpy.zeros((4, len(tau2_list)))

    for r in range(repeat):
        h_mat = ini_h_mat.copy()
        for n in range(n_devices):
            subcarrier_scale_list = numpy.zeros(K)
            subcarrier_scale_list[0:int(K / 4)] = 0.1 * numpy.random.random_sample(int(K / 4)) + 0.1
            subcarrier_scale_list[int(K / 4):int(K / 2)] = 0.2 * numpy.random.random_sample(int(K / 4)) + 0.2
            subcarrier_scale_list[int(K / 2):] = 1
            subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
            for k in range(K):
                h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]

        for i in range(len(number_of_devices_list)):
            print('iteration ' + str(r) + ' - number of devices: ' + str(number_of_devices_list[i]))
            active_devices = []
            random_permutation = numpy.random.permutation(n_devices)
            for perm_iter in range(number_of_devices_list[i]):
                active_devices.append(random_permutation[perm_iter])
            model.set_active_devices(active_devices)

            for j in range(len(tau2_list)):
                if j == 0:
                    mode = PURE
                else:
                    mode = GRAPH
                model.set_system_params(w_mat, h_mat, tau2_list[j], P, mode)
                results[j, r, i] = FashionMNIST_test(model, device, test_loader)
                stored_results[j, i] = results[j, r, i]
        out_file_name = home_dir + 'Outputs/PartialDevices_demo_' + data_name + '_num-range_' + str(
            number_of_devices_list[0]) + '-' + str(number_of_devices_list[-1]) + '_repeat_' + str(r) + '_results.npz'
        numpy.savez(out_file_name, res=stored_results)
    plot_results(results, number_of_devices_list, data_name, legends)

def CNN_versus_devices_demo():
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1800, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    img_size_list = [32, 32, 16, 16, 16, 16, 16, 16, 16, 16]
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = PartialDevicesSplitCNN(img_size_list, device=device).to(device)
    model_state_dict = torch.load('vertically_split_EuroSAT_n_devices_' + str(len(img_size_list)) + '.pt')
    model.load_state_dict(model_state_dict)

    n_devices = len(img_size_list)
    J = 128
    K = 128
    m = 5
    P = 10
    tau2_list = [0, 0.5, 1, 2]
    number_of_devices_list = [2, 4, 6, 8, 10]
    ini_h_mat = abs(numpy.random.randn(n_devices, K, m))
    w_mat = EuroSAT_tau_mat_processing(args, model, n_devices, J)

    repeat = 50
    data_name = 'EuroSAT_CNN'
    legends = [r"Proposed Approach: $\sigma^{2}=0$", r"Proposed Approach: $\sigma^{2}=0.5$",
               r"Proposed Approach: $\sigma^{2}=1$", r"Proposed Approach: $\sigma^{2}=2$"]
    results = numpy.zeros((4, repeat, len(number_of_devices_list)))
    # objectives = numpy.zeros((4, repeat, len(tau2_list)))
    stored_results = numpy.zeros((4, len(number_of_devices_list)))

    for r in range(repeat):
        h_mat = ini_h_mat.copy()
        for n in range(n_devices):
            subcarrier_scale_list = numpy.zeros(K)
            subcarrier_scale_list[0:int(K / 4)] = 0.1 * numpy.random.random_sample(int(K / 4)) + 0.1
            subcarrier_scale_list[int(K / 4):int(K / 2)] = 0.2 * numpy.random.random_sample(int(K / 4)) + 0.2
            subcarrier_scale_list[int(K / 2):] = 1
            subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
            for k in range(K):
                h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]

        for i in range(len(number_of_devices_list)):
            print('iteration ' + str(r) + ' - number of devices: ' + str(number_of_devices_list[i]))
            active_devices = []
            random_permutation = numpy.random.permutation(n_devices)
            for perm_iter in range(number_of_devices_list[i]):
                active_devices.append(random_permutation[perm_iter])
            model.set_active_devices(active_devices)

            for j in range(len(tau2_list)):
                if j == 0:
                    mode = PURE
                else:
                    mode = GRAPH
                model.set_system_params(w_mat, h_mat, tau2_list[j], P, mode)
                results[j, r, i] = EuroSAT_test(model, device, test_loader)
                stored_results[j, i] = results[j, r, i]
        out_file_name = home_dir + 'Outputs/PartialDevices_demo_' + data_name + '_num-range_' + str(
            number_of_devices_list[0]) + '-' + str(number_of_devices_list[-1]) + '_repeat_' + str(r) + '_results.npz'
        numpy.savez(out_file_name, res=stored_results)
    plot_results(results, number_of_devices_list, data_name, legends)

def ResNet_versus_devices_demo():
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

    img_size_list = [32, 32, 16, 16, 16, 16, 16, 16, 16, 16]
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = PartialDevicesSplitResnet(img_size_list, device=device).to(device)
    model_state_dict = torch.load('vertically_split_Resnet_n_devices_' + str(len(img_size_list)) + '.pt')
    model.load_state_dict(model_state_dict)

    # system parameters
    n_devices = len(img_size_list)
    J = 64
    K = 64
    m = 5
    P = 1
    tau2_list = [0, 0.5, 1, 2]
    number_of_devices_list = [2, 4, 6, 8, 10]
    ini_h_mat = numpy.random.rayleigh(1, size=(n_devices, K, m))
    w_mat = EuroSAT_tau_mat_processing(args, model, n_devices, J)

    repeat = 50
    data_name = 'EuroSAT_ResNet'
    legends = [r"Proposed Approach: $\sigma^{2}=0$", r"Proposed Approach: $\sigma^{2}=0.5$",
               r"Proposed Approach: $\sigma^{2}=1$", r"Proposed Approach: $\sigma^{2}=2$"]
    results = numpy.zeros((4, repeat, len(number_of_devices_list)))
    # objectives = numpy.zeros((4, repeat, len(tau2_list)))
    stored_results = numpy.zeros((4, len(number_of_devices_list)))

    for r in range(repeat):
        h_mat = ini_h_mat.copy()
        for n in range(n_devices):
            subcarrier_scale_list = numpy.zeros(K)
            subcarrier_scale_list[0:int(K / 4)] = 0.1 * numpy.random.random_sample(int(K / 4)) + 0.1
            subcarrier_scale_list[int(K / 4):int(K / 2)] = 0.2 * numpy.random.random_sample(int(K / 4)) + 0.2
            subcarrier_scale_list[int(K / 2):] = 1
            subcarrier_scale_list = subcarrier_scale_list[numpy.random.permutation(K)]
            for k in range(K):
                h_mat[n, k] = subcarrier_scale_list[k] * h_mat[n, k]

        for i in range(len(number_of_devices_list)):
            print('iteration ' + str(r) + ' - number of devices: ' + str(number_of_devices_list[i]))
            active_devices = []
            random_permutation = numpy.random.permutation(n_devices)
            for perm_iter in range(number_of_devices_list[i]):
                active_devices.append(random_permutation[perm_iter])
            model.set_active_devices(active_devices)

            for j in range(len(tau2_list)):
                if j == 0:
                    mode = PURE
                else:
                    mode = GRAPH
                model.set_system_params(w_mat, h_mat, tau2_list[j], P, mode)
                results[j, r, i] = EuroSAT_test(model, device, test_loader)
                stored_results[j, i] = results[j, r, i]
        out_file_name = home_dir + 'Outputs/PartialDevices_demo_' + data_name + '_num-range_' + str(
            number_of_devices_list[0]) + '-' + str(number_of_devices_list[-1]) + '_repeat_' + str(r) + '_results.npz'
        numpy.savez(out_file_name, res=stored_results)
    plot_results(results, number_of_devices_list, data_name, legends)

if __name__ == '__main__':
    MLP_versus_devices_demo()
    CNN_versus_devices_demo()
    ResNet_versus_devices_demo()