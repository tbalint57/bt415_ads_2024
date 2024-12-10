import matplotlib.pyplot as plt
import numpy as np


def plot_arrays(arrays, labels=None, colours=None, title=None, xlabel=None, ylabel=None):

    for i, array in enumerate(arrays):
        x = np.arange(len(array))
        
        label = labels[i] if labels else None
        color = colours[i] if colours else None

        plt.plot(x, array, label=label, color=color)

    if labels:
        plt.legend()

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True)
    plt.show()


def visualise_output_values(feature_arrays, feature_labels=None, title=None):
    print(len(feature_arrays), len(feature_labels))

    sorted_arrays = [np.sort(np.copy(feature_array)) for feature_array in feature_arrays]

    plot_arrays(sorted_arrays, labels=feature_labels, title=title, xlabel="n-th lowest value", ylabel="value")
