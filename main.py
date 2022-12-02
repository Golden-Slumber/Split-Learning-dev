# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import torch

from scipy import signal
import matplotlib.pyplot as plt
import numpy


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t = numpy.linspace(0, 0.25, 1000, False)  # 1 second
    sig11 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 0.2)
    sig12 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 3)
    sig13 = numpy.sin(2 * numpy.pi * 10 * t) + numpy.sin(2 * numpy.pi * 20 * t + 5)
    sig21 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 0.2)
    sig22 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 3)
    sig23 = numpy.sin(2 * numpy.pi * 20 * t) + numpy.sin(2 * numpy.pi * 15 * t + 5)
    sig31 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 5)
    sig32 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 0.2)
    sig33 = numpy.sin(2 * numpy.pi * 30 * t) + numpy.sin(2 * numpy.pi * 20 * t + 3)
    plt.plot(t, sig11+sig12+sig13+sig21+sig22+sig23+sig31+sig32+sig33, color='dimgrey', linewidth='8')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
