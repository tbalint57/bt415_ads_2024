import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from . import pandas_utils
import math
from scipy.spatial.distance import pdist, squareform


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
        plt.subplot(num_of_plots, 1, num_plotted)

        if feature_col == "OA":
            continue

        for goal_col in goal_df.columns:
            if goal_col == "OA":
                continue
                
            a, b = np.polyfit(df[feature_col], df[goal_col], 1)
            plt.plot(df[feature_col], a*df[feature_col]+b, label=goal_col)


        plt.xlabel(feature_col)
        plt.ylabel("Goal values")
        plt.title("Relationship between " + feature_col + " and the goal")

        num_plotted += 1


    plt.legend()
    plt.tight_layout()
    plt.show()


def visualise_relationship(df, column_a, column_b):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column_a], df[column_b], color="green", alpha=0.7)


    a, b = np.polyfit(df[column_a], df[column_b], 1)
    plt.plot(df[column_a], a*df[column_a]+b)

    plt.xlabel(column_a)
    plt.ylabel(column_b)
    plt.title("Relationship between " + column_a + " and " + column_b)


def visualise_feature_on_map(df, feature_name):
    feature_min = df[feature_name].min()
    feature_max = df[feature_name].max()
    
    normalized_feature = (df[feature_name] - feature_min) / (feature_max - feature_min)
    
    scatter = plt.scatter(
        df['lat'], df['long'], 
        c=normalized_feature, cmap='viridis', s=5, alpha=0.5
    )
    
    cbar = plt.colorbar(scatter, orientation="vertical")
    cbar.set_label(feature_name)
    cbar.set_ticks(np.linspace(0, 1, num=5))
    cbar.set_ticklabels([f"{feature_min:.2f}", 
                         f"{(feature_min + (feature_max - feature_min) * 0.25):.2f}",
                         f"{(feature_min + (feature_max - feature_min) * 0.5):.2f}",
                         f"{(feature_min + (feature_max - feature_min) * 0.75):.2f}",
                         f"{feature_max:.2f}"])
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Visualization of {feature_name} on Map")
    
    plt.show()


def plot_feature_on_map_relative_to_median(df, feature_name):
    feature_median = df[feature_name].median()
    feature_min = df[feature_name].min()
    feature_max = df[feature_name].max()
    
    difference_from_median = df[feature_name] - feature_median
    
    max_distance = max(abs(feature_min - feature_median), abs(feature_max - feature_median))
    normalized_difference = difference_from_median / max_distance
    
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    scatter = plt.scatter(
        df['long'], df['lat'], 
        c=normalized_difference, cmap=cmap, norm=norm, s=5, alpha=0.5
    )
    
    cbar = plt.colorbar(scatter, orientation="vertical")
    cbar.set_label(f"Deviation from Median ({feature_name})")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels([
        f"{feature_min:.2f} (Min)", 
        f"{0.5 * (feature_median + feature_min):.2f}",
        f"{feature_median:.2f} (Median)", 
        f"{0.5 * (feature_median + feature_median):.2f}",
        f"{feature_max:.2f} (Max)"
    ])
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Visualization of {feature_name} (Deviation from Median) on Map")
    
    plt.show()


# ----- ===== -----


def plot_values_increasing(features_df, plot_size=(6, 6), title="Sorted Feature Values"):
    plt.figure(figsize=plot_size)

    for column in features_df.columns:
        plt.plot(sorted(features_df[column]), label=column)

    plt.legend(title="Features")
    plt.xlabel("n-th lowest value")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_values_distribution(features_df, base_figsize=(6, 6), title="Values by Frequency"):
    rows = int(math.ceil(len(features_df.columns[2:]) / 3))
    cols = min(3, len(features_df.columns[2:]))
    
    plot_size_x = base_figsize[0] * cols
    plot_size_y = base_figsize[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(plot_size_x, plot_size_y), constrained_layout=True)

    axes = axes.flatten()

    for i, column in enumerate(features_df.columns):
        if i < len(axes):
            axes[i].hist(features_df[column], bins=50, alpha=0.7)
            median = features_df[column].median()  # Calculate the median
            axes[i].axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
            axes[i].set_title(f"{title} : {column}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            axes[i].legend()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)


    plt.title(title)
    plt.tight_layout()
    plt.show()
    

def plot_values_on_map_relative_to_median(features_df, loc=None, base_figsize=(6, 6)):
    filtered_df = features_df
    if loc is not None:
        lat, lon = loc
        filtered_df = pandas_utils.filter_by_cords(features_df, lat, lon, size_km=10)
    
    rows = int(math.ceil(len(features_df.columns[2:]) / 3))
    cols = min(3, len(features_df.columns[2:]))
    
    plot_size_x = base_figsize[0] * cols
    plot_size_y = base_figsize[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(plot_size_x, plot_size_y), constrained_layout=True)

    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    feature_names = features_df.columns.difference(["OA", "lat", "long"])

    for i, feature_name in enumerate(feature_names):
        if i < len(axes):  # Ensure we don't exceed the number of available subplots
            feature_median = features_df[feature_name].median()
            feature_min = features_df[feature_name].min()
            feature_max = features_df[feature_name].max()

            difference_from_median = filtered_df[feature_name] - feature_median
            max_distance = max(abs(feature_min - feature_median), abs(feature_max - feature_median))
            normalized_difference = difference_from_median / max_distance

            cmap = plt.get_cmap('coolwarm')
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

            scatter = axes[i].scatter(
                filtered_df['long'], filtered_df['lat'], 
                c=normalized_difference, cmap=cmap, norm=norm, s=5, alpha=0.5
            )

            # Associate colorbar explicitly with the subplot
            cbar = fig.colorbar(scatter, ax=axes[i], orientation="vertical")
            cbar.set_label(f"Deviation from Median ({feature_name})")
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_ticklabels([
                f"{feature_min:.2f} (Min)", 
                f"{0.5 * (feature_median + feature_min):.2f}",
                f"{feature_median:.2f} (Median)", 
                f"{0.5 * (feature_median + feature_max):.2f}",
                f"{feature_max:.2f} (Max)"
            ])

            axes[i].set_title(feature_name)
            axes[i].set_xlabel("Longitude")
            axes[i].set_ylabel("Latitude")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.show()


def plot_oa_clusters(df, plot_size=(5, 5)):
    plt.figure(figsize=plot_size)
    for cluster in df["cluster"].unique():
        cluster_data = df[df["cluster"] == cluster]
        plt.scatter(
            cluster_data["long"], cluster_data["lat"],
            label=f"Cluster {cluster}", s=5, alpha=0.7
        )
            
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title("Clustered Locations")
    plt.legend()
    plt.show()


def plot_difference_matrix_for_features(df, plot_size=(10, 10)):
    # Transpose the DataFrame to compute pairwise distances between features
    features_matrix = df.T

    # Compute the distance matrix using Euclidean distance
    distance_matrix = squareform(pdist(features_matrix, metric="euclidean"))

    # Feature names
    feature_names = df.columns

    # Plot the distance matrix as a heatmap
    plt.figure(figsize=plot_size)
    plt.imshow(distance_matrix, interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Euclidean Distance")

    plt.title("Feature Distance Matrix")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)

    plt.tight_layout()
    plt.show()