import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

home_dir = '../'
sys.path.append(home_dir)


def plot_results(res, number_of_devices_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(number_of_devices_list, numpy.median(res[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(number_of_devices_list, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xlabel('Number of Devices', fontsize=25)
    # plt.ylabel('Inference Accuracy', fontsize=25)
    plt.xlabel('参与设备数量', fontsize=25)
    plt.ylabel('推理准确率', fontsize=25)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_accuracy_total.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    repeat = 10
    tau2_list = [0, 1, 2, 4]
    # tau2_list = [0, 1, 2.2, 4]
    number_of_devices_list = [2, 4, 6, 8, 10, 12, 14]

    data_name = 'fashionMNIST'
    # data_name = 'cifar10'
    # legends = ['Scheme 1', r"Scheme 2: $\sigma^{2}=0$", r"Scheme 2: $\sigma^{2}=1$", r"Scheme 2: $\sigma^{2}=2$",
    #            r"Scheme 2: $\sigma^{2}=4$"]
    legends = ['无噪声', r"所提出方法: $\sigma^{2}=0$", r"所提出方法: $\sigma^{2}=1$", r"所提出方法: $\sigma^{2}=2$",
               r"所提出方法: $\sigma^{2}=4$"]
    # legends = ['Scheme 1', r"Scheme 2: $\sigma^{2}=0$", r"Scheme 2: $\sigma^{2}=1$", r"Scheme 2: $\sigma^{2}=2.2$"]
    results = numpy.zeros((5, repeat, len(number_of_devices_list)))

    for i in range(len(tau2_list)):
        if i == 1:
            for r in range(repeat):
                if r <= 14:
                    out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_repeat_' + str(
                        r) + '_partial_results.npz'
                else:
                    out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(
                        tau2_list[i]) + '_repeat_' + str(r) + '_partial_results.npz'
                npz_file = numpy.load(out_file_name, allow_pickle=True)
                results[0][r] = npz_file['res'][1]
                results[2][r] = npz_file['res'][0]
        else:
            for r in range(repeat):
                out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(
                    tau2_list[i]) + '_repeat_' + str(r) + '_partial_results.npz'
                npz_file = numpy.load(out_file_name, allow_pickle=True)
                results[i + 1][r] = npz_file['res'][0]
        # for r in range(repeat):
        #     out_file_name = home_dir + 'Outputs/number_of_devices_demo_' + data_name + '_tau2_' + str(
        #         tau2_list[i]) + '_repeat_' + str(r) + '_partial_results.npz'
        #     npz_file = numpy.load(out_file_name, allow_pickle=True)
        #     results[0][r] = npz_file['res'][1]
        #     results[i + 1][r] = npz_file['res'][0]

    plot_results(results, number_of_devices_list, data_name, legends)
