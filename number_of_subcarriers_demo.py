import sys
import numpy
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *
from multi_modality_nural_network import MultiModalityNet
from Hungarian_based_system_optimization import graph_based_alternating_optimization_framework
from split_inference import WirelessSplitNet, train, test, split_layer, get_num_neurons, FashionMNIST_training, \
    tau_mat_processing

home_dir = './'
sys.path.append(home_dir)


def plot_results(res, obj, K_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(K_list, numpy.median(res[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xticks(K_list)
    plt.xlabel('Number of Subcarriers', fontsize=20)
    plt.ylabel('Inference Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/number_of_subcarriers_demo_' + data_name + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(K_list, numpy.median(obj[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=15)
    plt.xticks(K_list)
    plt.xlabel('Number of Subcarriers', fontsize=20)
    plt.ylabel('Objective Value', fontsize=20)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/number_of_subcarriers_demo_' + data_name + '_objective.pdf'
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

    train_flag = False
    if train_flag:
        FashionMNIST_training(args, n_devices, next_layer_neurons, pre_layer_neurons)
    else:
        # load model
        model_state_dict = torch.load('vertically_split_fashionmnist_n_devices_' + str(n_devices) + '.pt')
        model = WirelessSplitNet(next_layer_neurons, pre_layer_neurons, device=device).to(device)
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
        m = 5
        P = 10
        # K_list = [32, 40, 48, 56, 64, 72, 80, 88, 96]
        K_list = [32, 48, 64, 80, 96, 112, 128]
        tau2_list = [0.25, 1, 2]

        ini_h_mat = abs(numpy.random.randn(n_devices, K_list[-1], m))
        for n in range(n_devices):
            subcarrier_scale_list = numpy.ones(K_list[-1])
            subcarrier_scale_list[0:int(K_list[-1] / 4)] = 0.1 * numpy.random.random_sample(int(K_list[-1] / 4)) + 0.1
            for k in range(K_list[-1]):
                ini_h_mat[n, k] = subcarrier_scale_list[k] * ini_h_mat[n, k]

        w_mat = tau_mat_processing(args, model, n_devices, J)
        repeat = 50
        data_name = 'fashionMNIST'
        legends = [r"$\sigma^{2}=0.25$", r"$\sigma^{2}=1$", r"$\sigma^{2}=2$"]
        results = numpy.zeros((len(tau2_list), repeat, len(K_list)))
        objectives = numpy.zeros((len(tau2_list), repeat, len(K_list)))
        stored_results = numpy.zeros((len(tau2_list), len(K_list)))
        stored_objectives = numpy.zeros((len(tau2_list), len(K_list)))

        for r in range(repeat):
            for i in range(len(K_list)):

                h_mat = numpy.zeros((n_devices, K_list[i], m))
                for n in range(n_devices):
                    tmp_h_mat = ini_h_mat.copy()[n]
                    tmp_h_mat = tmp_h_mat[numpy.random.permutation(K_list[-1])]
                    tmp_h_mat = tmp_h_mat[0:K_list[i]]
                    h_mat[n] = tmp_h_mat

                for l in range(len(tau2_list)):
                    print('iteration ' + str(r) + '- number of subcarriers: ' + str(
                        K_list[i]) + ' - noise variance: ' + str(tau2_list[l]))
                    objectives[l, r, i] = model.set_system_params(w_mat, h_mat, tau2_list[l], P, GRAPH)
                    stored_objectives[l, i] = objectives[l, r, i]
                    results[l, r, i] = test(model, device, test_loader)
                    stored_results[l, i] = results[l, r, i]

            out_file_name = home_dir + 'Outputs/number_of_subcarriers_demo_' + data_name + '_nsubcarrier-range_' + str(
                K_list[0]) + '-' + str(K_list[-1]) + '_repeat_' + str(r) + '_results.npz'
            numpy.savez(out_file_name, res=stored_results, obj=stored_objectives)
        plot_results(results, objectives, K_list, data_name, legends)
