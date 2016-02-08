from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def load():
    return np.loadtxt('Sample_Data.csv', delimiter=',')

split_data = lambda data: (data[:,:-3], data[:,-3:])

def plot_data(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    gcamp_data, locomotion_data = split_data(data)

    ax1.plot(locomotion_data)
    ax1.set_title('locomotion data')

    ax2.plot(gcamp_data)
    ax2.set_title('gcamp data')

if __name__ == "__main__":
    data = load()
    plot_data(data[:2000])
    plt.show()
