from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

split_data = lambda data: (data[:,:-3], data[:,-3:])

def load():
    return np.loadtxt('Sample_Data.csv', delimiter=',')

def plot_data(*args):
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    rc('text', usetex=True)

    gcamp_data, locomotion_data = split_data(*args) if len(args) == 1 else args

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), dpi=200)

    lines = ax1.plot(locomotion_data)
    ax1.set_title(r'locomotion data')
    plt.legend(iter(lines), [r'$v_{\text{side}}$', r'$v_{\text{forward}}$', r'$v_{\text{angle}}$'],
               loc='lower left', ncol=3)

    lines = ax2.plot(gcamp_data)
    ax2.set_title(r'gcamp data')
    plt.legend(iter(lines), [r'$\gamma_{}$'.format(i) for i in range(1, 5)],
               loc='lower left', ncol=4)

if __name__ == "__main__":
    data = load()
    plot_data(data[:2000])
    plt.show()
