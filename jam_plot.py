"""
Plotting of jam spectrums with bin#+counts as file format
"""


import matplotlib.pyplot as plt
import seaborn as sbs
import numpy as np

sbs.set_context('poster')


def jam_read(filename):
    bins = []
    counts = []
    with open(filename,'r') as f:
        for line in f:
            line = line.split()
            bins.append(float(line[0]))
            counts.append(float(line[1]))
    return counts,bins
    
def jam_plot(counts,bins):
    plt.hist(bins,bins=bins,weights=counts)
    plt.show()

