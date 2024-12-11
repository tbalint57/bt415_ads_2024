import matplotlib.pyplot as plt
import numpy as np


def plot_arrays(arrays, labels=None, colours=None, title=None, xlabel=None, ylabel=None):

    for i, array in enumerate(arrays):
        x = np.arange(len(array))
        
        label = labels[i] if labels is not None else None
        color = colours[i] if colours is not None else None

        plt.plot(x, array, label=label, color=color)

    if labels is not None:
        plt.legend()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.grid(True)


def visualise_feature_values_increasing(features_df, feature_labels=None, title=None):
    if not feature_labels:
        feature_labels = features_df.columns

    sorted_feature_arrays = [np.sort(feature_array) for feature_array in np.transpose(features_df.to_numpy())]

    plot_arrays(sorted_feature_arrays, labels=feature_labels, title=title, xlabel="n-th lowest value", ylabel="value")


def visualise_feature_outliers(features_df, num_outliers=500, feature_labels=None, title=None):
    if not feature_labels:
        feature_labels = features_df.columns

    sorted_feature_arrays = [np.sort(feature_array) for feature_array in np.transpose(features_df.to_numpy())]

    sorted_feature_arrays_mins = [feature_array[:num_outliers] for feature_array in sorted_feature_arrays]
    sorted_feature_arrays_maxs = [feature_array[-num_outliers:] for feature_array in sorted_feature_arrays]

    plt.subplot(1, 2, 1)
    plot_arrays(sorted_feature_arrays_mins, labels=feature_labels, title=title, xlabel="n-th lowest value", ylabel="value")
    plt.subplot(1, 2, 2)
    plot_arrays(sorted_feature_arrays_maxs, labels=feature_labels, title=title, xlabel="n-th highest value", ylabel="value")
