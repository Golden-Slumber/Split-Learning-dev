import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

home_dir = '../'
sys.path.append(home_dir)


def plot_results(res, obj, tau2_list, data_name, legends, legendsV2, idx):
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
    plt.xlabel('Noise Variance', fontsize=20)
    plt.ylabel('Inference Accuracy', fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/versus_noise_' + data_name + '_accuracy_' + str(idx) + '.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    for i in range(len(legendsV2)):
        line, = plt.plot(tau2_list, numpy.median(obj[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legendsV2, fontsize=15)
    plt.xticks(tau2_list)
    # plt.xlabel(r"$\sigma^{2}$", fontsize=20)
    plt.xlabel('Noise Variance', fontsize=20)
    plt.ylabel('Objective Value', fontsize=20)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/versus_noise_' + data_name + '_objective_' + str(idx) + '.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    repeat = 7
    data_name = 'cifar10'
    # tau2_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    tau2_list = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
    legends = ['Scheme 1', 'Scheme 2', 'Scheme 3']
    legendsV2 = ['Scheme 1', 'Scheme 2']

    for i in range(1):
        results = numpy.zeros((3, repeat, len(tau2_list)))
        objectives = numpy.zeros((3, repeat, len(tau2_list)))
        stored_results = numpy.zeros((3, len(tau2_list)))
        stored_objectives = numpy.zeros((3, len(tau2_list)))

        for j in range(repeat):
            file_name = home_dir + 'Outputs/aircomp_based_split_inference_demo_' + data_name + '_repeat_' + str(
                i * repeat + j) + '_partial_results_v2.npz'
            npz_file = numpy.load(file_name, allow_pickle=True)
            stored_results = npz_file['res']
            stored_objectives = npz_file['obj']

            for k in range(3):
                for tau_idx in range(len(tau2_list)):
                    results[k, j, tau_idx] = stored_results[k, tau_idx]
                    objectives[k, j, tau_idx] = stored_objectives[k, tau_idx]

        plot_results(results, objectives, tau2_list, data_name, legends, legendsV2, i)
