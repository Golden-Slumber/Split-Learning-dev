# This is a sample Python script.
import matplotlib
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import torch
import sys
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy

from constants import *
home_dir = './'
sys.path.append(home_dir)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
    plt.ylim(50, 400)
    plt.ylabel('Objective Value', fontsize=20)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/aircomp_based_inference_' + data_name + '_objective.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # t = numpy.linspace(0, 0.25, 1000, False)  # 1 second
    # sig11 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 0.2)
    # sig12 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 3)
    # sig13 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 5)
    # sig21 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 0.2)
    # sig22 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 3)
    # sig23 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 5)
    # sig31 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 5)
    # sig32 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 0.2)
    # sig33 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 3)
    # plt.plot(t, sig11+sig12+sig13+sig21+sig22+sig23+sig31+sig32+sig33, color='dimgrey', linewidth='8')
    # plt.show()

    # data_name = 'fashionMNIST'
    # # data_name = 'cifar10'
    # legends = ['Scheme 1', 'Scheme 2']
    # repeat = 1
    # tau2_list = [0.02, 0.22, 0.42, 0.62, 0.82]
    # results = numpy.zeros((2, repeat, len(tau2_list)))
    #
    # scheme1 = [154.42208675345827, 316.0439205164859, 331.08136655992365, 333.7328386683281, 333.13249448939393]
    # scheme2 = [74.73877607398234, 78.39302382216226, 81.96748537173329, 85.4647381796685, 88.88724977761848]
    #
    # for r in range(repeat):
    #     for i in range(len(tau2_list)):
    #         results[0, r, i] = scheme1[i]
    #         results[1, r, i] = scheme2[i]
    #
    # plot_results(results, tau2_list, data_name, legends)
    out_file_name = home_dir + 'Outputs/aircomp_based_split_inference_' + 'fashionMNIST' + '_nvar-range_' + str(
        0.25) + '-' + str(2) + '_repeat_' + str(0) + '_results.npz'

    tmp = numpy.load(out_file_name)
    key = tmp.files[0]
    print(tmp[key])
    print(tmp.files)
    perm = numpy.random.permutation(5)
    print(perm)
    print(perm[2])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
