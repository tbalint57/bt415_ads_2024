from .config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from .utils import aws_utils, pandas_utils, plot_utils


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

    return poi_count


def cluster_locations(df, n_clusters=3):
    """
    Cluster locations based on features using KMeans
    """
    feature_columns = df.columns.difference(["location", "coordinates"])
    X = df[feature_columns]

    kmeans = KMeans(n_clusters=n_clusters)

    clustered_df = df.copy()
    clustered_df["cluster"] = kmeans.fit_predict(X)

    return clustered_df


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


def normalise_data_frame(df):
    """
    Normalise data frame
    """
    normalised_df = df.copy()

    feature_columns = df.columns.difference(["location", "coordinates"])
    numerical_df = df[feature_columns]

    normalised_features = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())
    
    normalised_df[numerical_df.columns] = normalised_features

    return normalised_df


def get_sales_in_region(conn, latitude, longitude, from_year, distance_km = 1):
    from_date = str(from_year) + "-01-01"
    distance_degree = distance_km / 111

    cur = conn.cursor()
    cur.execute(f'''
    SELECT pc.price AS price, 
           pc.date_of_transfer AS date_of_transfer, 
           pc.postcode AS postcode, 
           pp.street AS street, 
           pp.primary_addressable_object_name AS number
    FROM prices_coordinates_data AS pc
    INNER JOIN pp_data AS pp 
    ON pp.postcode = pc.postcode 
        AND pp.price = pc.price 
        AND pp.date_of_transfer = pc.date_of_transfer
    WHERE pc.date_of_transfer >= "{from_date}" 
        AND pc.latitude BETWEEN {latitude} - {distance_degree} AND {latitude} + {distance_degree}
        AND pc.longitude BETWEEN {longitude} - {distance_degree} AND {longitude} + {distance_degree}
    ''')

    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(result, columns=columns)


def get_building_addresses_in_region(latitude, longitude, distance_km=1):
    buildings = access.query_osm(latitude, longitude, {"building": True}, distance_km)
    addresses = buildings[["addr:postcode", "addr:street", "addr:housenumber", "building", "geometry"]].dropna().rename(columns={"addr:postcode": "postcode", "addr:street": "street", "addr:housenumber": "number"})
    addresses["area"] = addresses["geometry"].area * 111 * 111 * 1000 * 1000
    addresses = addresses[["postcode", "street", "number", "building", "area"]]
    addresses["street"] = addresses["street"].apply(str.upper)
    return addresses




def visualise_relationship_for_field(features_df, field_name, goal_df, merge_on=["OA"]):
    feature_df = features_df[merge_on + [field_name]]
    df = pd.merge(feature_df, goal_df, on=merge_on)

    for goal_col in goal_df.columns:
        if goal_col in merge_on:
            continue
            
        a, b = np.polyfit(df[field_name], df[goal_col], 1)
        plt.plot(df[field_name], a*df[field_name]+b, label=goal_col)

    plt.xlabel(field_name)
    plt.ylabel("Goal values")
    plt.title("Relationship between " + field_name + " and the mode of transportation")

    plt.legend() 
    plt.show() 


def calculate_relationship_for_field(features_df, field_name, goal_df, merge_on=["OA"]):
    feature_df = features_df[merge_on + [field_name]]
    df = pd.merge(feature_df, goal_df, on=merge_on)

    avg, mx = 0, 0 

    for goal_col in goal_df.columns:
        if goal_col in merge_on:
            continue
            
        a, b = np.polyfit(df[field_name], df[goal_col], 1)
        avg += a
        mx = max(mx, abs(a))
    
    avg /= len(goal_df.columns) - 1

    return avg, mx


def compare_single_fields(feature_df, goal_df, input_col, goal_col, merge_on=["OA"]):
    df = pd.merge(feature_df, goal_df, on=merge_on)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(df[input_col], df[goal_col], alpha=0.7)
    plt.title(f'{input_col} vs {goal_col}')
    plt.xlabel(input_col)
    plt.ylabel(goal_col)
    plt.grid(True)
    plt.show()


def visualise_transport_data(conn):
    field_names = ["TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
    transport_df = pandas_utils.normalise_data_frame(aws_utils.query_AWS_load_table(conn, "census_data", field_names))
    plt.figure(figsize=(18, 6))
    plot_utils.visualise_feature_values_increasing(transport_df)


def visualise_transport_data_outliers(conn, number_of_outiers=500):
    transport_field_names = ["TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
    transport_df = pandas_utils.normalise_data_frame(aws_utils.query_AWS_load_table(conn, "census_data", transport_field_names))
    plt.figure(figsize=(18, 6))
    plot_utils.visualise_feature_outliers(transport_df, number_of_outiers)


def visualise_transport_and_age(conn):
    age_field_names = ["TS007_4_minus", "TS007_5_to_9", "TS007_10_to_15", "TS007_16_to_19", "TS007_20_to_24", "TS007_25_to_34", "TS007_35_to_49", "TS007_50_to_64", "TS007_65_to_74", "TS007_75_to_84", "TS007_85_plus"]
    transport_field_names = ["TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
    response_df = aws_utils.query_AWS_load_table(conn, "census_data", ["OA"] + transport_field_names + age_field_names)

    age_df = pandas_utils.normalise_data_frame(response_df[["OA"] + age_field_names], ["OA"])
    transpost_df = pandas_utils.normalise_data_frame(response_df[["OA"] + transport_field_names], ["OA"])

    plt.figure(figsize=(9, 33))
    plot_utils.visualise_relationship_by_components(age_df, transpost_df)


def visualise_car_usage_on_map(conn):
    transport_field_names = ["long", "lat", "TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
    response_df = aws_utils.query_AWS_load_table(conn, "census_data", transport_field_names)
    car_df = pandas_utils.normalise_data_frame(response_df, ["long", "lat"])[["long", "lat", "TS061_car_driving"]]

    plt.figure(figsize=(12, 12))
    plot_utils.visualise_feature_on_map_relative_to_median(car_df, "TS061_car_driving")


def visualise_transport_usages_on_map(conn):
    transport_field_names = ["long", "lat", "TS061_underground_tram", "TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
    response_df = aws_utils.query_AWS_load_table(conn, "census_data", transport_field_names)
    response_df = pandas_utils.normalise_data_frame(response_df, ["long", "lat"])

    plt.figure(figsize=(12, 12))
    for feature_name in transport_field_names:
        plot_utils.visualise_feature_on_map_relative_to_median(response_df, feature_name)


    
