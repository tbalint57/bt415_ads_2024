from .config import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {"amenity": True, "tourism": True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    pois = access.query_osm(latitude, longitude, tags, distance_km)

    pois_df = pd.DataFrame(pois)

    poi_count = {}
    for tag in tags:
        poi_count[tag] = 0

        if tag in pois_df.columns:
            poi_count[tag] = pois_df[tag].notnull().sum()

    poi_count["amenity"] = pois_df["amenity"].isin(tags["amenity"]).sum()

    for amenity in tags["amenity"]:
        poi_count[amenity] = (pois_df["amenity"] == amenity).sum()

    return poi_count


def cluster_locations(df, n_clusters=3):
    """
    Cluster locations based on features using KMeans
    """
    feature_columns = df.columns.difference(["location", "coordinates"])
    X = df[feature_columns]

    kmeans = KMeans(n_clusters=n_clusters)
    df["cluster"] = kmeans.fit_predict(X)

    return df


def plot_clusters(df):
    """
    Plots locations based on their clusters.
    """
    df["latitude"] = df["coordinates"].apply(lambda x: x[0])
    df["longitude"] = df["coordinates"].apply(lambda x: x[1])
    
    plt.figure(figsize=(10, 8))
    for cluster in df["cluster"].unique():
        cluster_data = df[df["cluster"] == cluster]
        plt.scatter(
            cluster_data["longitude"], cluster_data["latitude"],
            label=f"Cluster {cluster}", s=50, alpha=0.6
        )
        
        for _, row in cluster_data.iterrows():
            plt.text(
                row["longitude"], row["latitude"], row["location"],
                fontsize=8, ha="right"
            )
            
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clustered Locations")
    plt.legend()
    plt.grid(True)
    plt.show()
