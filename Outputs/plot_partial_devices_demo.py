import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from constants import *

home_dir = '../'
sys.path.append(home_dir)


def plot_results(results, number_of_devices_list, data_name, legends):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    line_list = []
    for i in range(len(legends)):
        line, = plt.plot(number_of_devices_list, numpy.median(results[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=9, markeredgewidth=3, linewidth=2.5, markevery=1)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(number_of_devices_list, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Number of Participating Devices', fontsize=25)
    plt.ylabel('Inference Accuracy', fontsize=25)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/Plot_PartialDevices_demo_' + data_name + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()



if __name__ == '__main__':
    repeat = 50
    data_list = ['fashionMNIST', 'EuroSAT_CNN', 'EuroSAT_ResNet']
    legends = [r"Proposed Approach: $\sigma^{2}=0$", r"Proposed Approach: $\sigma^{2}=0.5$",
               r"Proposed Approach: $\sigma^{2}=1$", r"Proposed Approach: $\sigma^{2}=2$"]

    for data_name in data_list:
        if data_name == 'fashionMNIST':
            number_of_devices_list = [2, 4, 6, 8, 10, 12, 14]
        else:
            number_of_devices_list = [2, 4, 6, 8, 10]
        results = numpy.zeros((4, repeat, len(number_of_devices_list)))
        for r in range(repeat):
            out_file_name = home_dir + 'Outputs/PartialDevices_demo_' + data_name + '_num-range_' + str(
                number_of_devices_list[0]) + '-' + str(number_of_devices_list[-1]) + '_repeat_' + str(
                r) + '_results.npz'
            npz_file = numpy.load(out_file_name, allow_pickle=True)
            stored_results = npz_file['res']
            for k in range(4):
                for n in range(len(number_of_devices_list)):
                    results[k, r, n] = stored_results[k, n]
        plot_results(results, number_of_devices_list, data_name, legends)


