import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def visualise_relationship_by_components(feature_df, goal_df, merge_on=["OA"]):
    df = pd.merge(feature_df, goal_df, on=merge_on)

    num_of_plots = len(feature_df.columns) - 1
    num_plotted = 1

    for feature_col in feature_df.columns:

        if feature_col == "OA":
            continue

        for goal_col in goal_df.columns:
            if goal_col == "OA":
                continue
                
            a, b = np.polyfit(df[feature_col], df[goal_col], 1)
            plt.plot(df[feature_col], a*df[feature_col]+b, label=goal_col)

        plt.subplot(1, num_of_plots, num_plotted)

        plt.xlabel(feature_col)
        plt.ylabel("Goal values")
        plt.title("Relationship between " + feature_col + " and the goal")

        plt.legend()

        num_plotted += 1
    
    plt.show()


def visualise_relationship(df, column_a, column_b):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column_a], df[column_b], color="green", alpha=0.7)


    a, b = np.polyfit(df[column_a], df[column_b], 1)
    plt.plot(df[column_a], a*df[column_a]+b)

    plt.xlabel(column_a)
    plt.ylabel(column_b)
    plt.title("Relationship between " + column_a + " and " + column_b)