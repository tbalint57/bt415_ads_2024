from .config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from .utils import aws_utils, pandas_utils, plot_utils
import math
import json


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
                row["latitude"], row["longitude"], row["location"],
                fontsize=8, ha="right"
            )
            
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
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


# ----- ===== -----


def sample_census_data(conn, code, limit=3):
    columns = ["OA", "lat", "long"] + access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns, limit=limit)
    return census_df


# Functions to visualise the census data
def visualise_census_data_values(conn, code, columns=None, size=5):
    if columns is None:
        columns = access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_increasing(census_df, title="Value Set of Trensport Data", plot_size=(size, size))


def visualise_census_data_distribution(conn, code, columns=None, size=5):
    if columns is None:
        columns = access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_distribution(census_df, base_figsize=(size, size))


def visualise_census_by_distance_from_median_on_map(conn, code, columns=None, size=5):
    if columns is None:
        columns = access.get_census_data_column_names()[code]
    columns = ["lat", "long"] +  columns
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_on_map_relative_to_median(census_df, base_figsize=(size, size))


def visualise_census_data_locally(conn, locations, code, columns=None, size=5):
    if columns is None:
        columns = access.get_census_data_column_names()[code]
    columns = ["lat", "long"] +  columns
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    for location, name in locations.items():
        lat, lon = location
        print(name)
        plot_utils.plot_values_on_map_relative_to_median(census_df, loc=(lat, lon), base_figsize=(size, size), max_col_size=6, labels_on=False)


def cluster_oas(df, n_clusters):
    """
    Cluster locations based on features using KMeans
    """
    feature_columns = df.columns.difference(["OA", "lat", "long"])
    X = df[feature_columns]

    kmeans = KMeans(n_clusters=n_clusters)

    clustered_df = df.copy()
    clustered_df["cluster"] = kmeans.fit_predict(X)

    return clustered_df


def visualise_census_by_clustering(conn, code, n_clusters=5, columns=None, size=10):
    if columns is None:
        columns = ["lat", "long"] + access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    cluster_df = cluster_oas(census_df, n_clusters=n_clusters)
    plot_utils.plot_oa_clusters(cluster_df, plot_size=(size, size))

    london_loc = (51.513257, -0.098277)
    plot_utils.plot_oa_clusters(cluster_df, plot_size=(size, size), loc=london_loc, size_km=100)


def visualise_census_tables_by_difference_matrix(conn, code_a, code_b, size=10):
    columns_a = access.get_census_data_column_names()[code_a]
    columns_b = access.get_census_data_column_names()[code_b]

    columns =  columns_a + columns_b
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    census_df_a = census_df[columns_a]
    census_df_b = census_df[columns_b]

    plot_utils.plot_difference_matrix_between_features(census_df_a, census_df_b, plot_size=(size, size))


def visualise_census_tables_by_correlation(conn, code_a, code_b, size=10):
    columns_a = access.get_census_data_column_names()[code_a]
    columns_b = access.get_census_data_column_names()[code_b]

    columns =  columns_a + columns_b
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    census_df_a = census_df[columns_a]
    census_df_b = census_df[columns_b]

    plot_utils.plot_correlation_heatmap_between_features(census_df_a, census_df_b, plot_size=(size, size)) 


def visualise_census_joint_distribution(conn, column_a, column_b, size=10):
    columns =  [column_a, column_b]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    census_df_a = census_df[[column_a]]
    census_df_b = census_df[[column_b]]

    plot_utils.plot_hexbin_joint_distribution(census_df_a, census_df_b, plot_size=(size, size)) 


def visualise_census_joint_distribution_for_feature_table(conn, column_a, code, size=10):
    columns_b = access.get_census_data_column_names()[code]
    columns =  [column_a] + columns_b
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    census_df_a = census_df[[column_a]]
    census_df_b = census_df[columns_b]

    plot_utils.plot_fitted_lines(census_df_a, census_df_b, plot_size=(size, size)) 


def calculate_census_correlation(conn, code, save_file=None):
    if save_file and os.path.exists(save_file):
        with open(save_file, 'r') as f:
            return json.load(f)

    tables = access.get_census_data_column_names()

    if code == "density":
        # For handling querying for density
        tables["density"] = ["density"]

    correlation_values = {attribute: [] for attribute in tables[code]}
    goal_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", tables[code])


    for table_name, feature_columns in tables.items():
        if table_name == code:
            continue

        feature_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", tables[table_name])
        print(table_name)

        correlation_matrix = np.corrcoef(feature_df.T, goal_df.T)[:feature_df.shape[1], feature_df.shape[1]:]

        for i, feature_column in enumerate(feature_columns):
            for j, goal_column in enumerate(tables[code]):
                correlation = correlation_matrix[i, j]
                if not math.isnan(correlation): # That pesky "0 person household" attribute...
                    correlation_values[goal_column].append((correlation, feature_column))

    if save_file:
        with open(save_file, "w") as json_file:
            json.dump(correlation_values, json_file)

    return correlation_values


def get_strong_correlations(correlation_values, min_corr=0.7):
    best_correlation_values = {}
    avg_correlations = {}

    for goal_column, correlations in correlation_values.items():
        # Filter correlations based on absolute value >= 0.7
        filtered_correlations = [(corr, feature) for corr, feature in correlations if abs(corr) >= min_corr]
        best_correlation_values[goal_column] = filtered_correlations

        # Calculate the average correlation for filtered correlations, if any
        for correlation_value, feature_name in correlation_values[goal_column]:
            if feature_name not in avg_correlations:
                avg_correlations[feature_name] = []
            avg_correlations[feature_name].append(abs(correlation_value))

    return best_correlation_values


def get_census_correlations_better_than_target(correlation_values, target_correlations, min_correlation_diff=0):
    target_correlations_dict = {key: value for (value, key) in target_correlations}
    better_than_target_correlations = {}

    for key in correlation_values:
        better_than_target_correlations[key] = []
        for correlation_value, feature_name in correlation_values[key]:
            if feature_name in target_correlations_dict and abs(correlation_value) >+ abs(target_correlations_dict[feature_name]) + min_correlation_diff  and abs(correlation_value) < 0.98:
                better_than_target_correlations[key].append(feature_name)
    
    return better_than_target_correlations


def visualise_correlations(correlation_dict):
    data = []

    for category, values in correlation_dict.items():
        for correlation, label in values:
            data.append({
                "Category": category,
                "Variable": label,
                "Correlation": correlation
            })

    if not data:
        print("No data to visualize.")
        return pd.DataFrame()

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame by Category and Correlation (optional)
    df = df.sort_values(by=["Category", "Correlation"], ascending=[True, False])

    return df


def visualise_census_feture_against_density(conn, feature, size=10):
    columns = ["lat", "long", feature, "density"]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)

    plot_utils.plot_values_on_map_relative_to_median(census_df, base_figsize=(size/2, size/2))




# Functions to visualise the osm data
def sample_osm_data(conn, type, limit=3):
    osm_df = aws_utils.query_AWS_load_table(conn, type, columns=None, limit=limit)
    return osm_df


def visualise_osm_data_distribution(conn, type, size=3):
    osm_df = aws_utils.query_AWS_load_table(conn, type).drop(columns=["OA", "lat", "long"])
    plot_utils.plot_values_distribution(osm_df, base_figsize=(size, size))


def visualise_osm_by_distance_from_median_on_map(conn, type, size=3):
    osm_df = aws_utils.query_AWS_load_table(conn, type).drop(columns=["OA"])
    plot_utils.plot_values_on_map_relative_to_median(osm_df, base_figsize=(size, size))


def visualise_osm_data_locally(conn, locations, type, size=3):
    census_df = aws_utils.query_AWS_load_table(conn, type).drop(columns=["OA"])
    for location, name in locations.items():
        lat, lon = location
        print(name)
        plot_utils.plot_values_on_map_relative_to_median(census_df, loc=(lat, lon), base_figsize=(size, size), max_col_size=6, labels_on=False)


def calculate_osm_correlation(conn, type, code, save_file=None):
    if save_file and os.path.exists(save_file):
        with open(save_file, 'r') as f:
            return json.load(f)

    tables = access.get_census_data_column_names()
    tables["density"] = ["density"]

    correlation_values = {attribute: [] for attribute in tables[code]}

    goal_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", tables[code] + ["OA"])
    feature_df = aws_utils.query_AWS_load_table(conn, type).drop(columns=["lat", "long"])

    goal_columns = tables[code]
    feature_columns = feature_df.columns.drop("OA")

    joined_df = goal_df.merge(feature_df, how="inner", on=["OA"])

    goal_df = joined_df[goal_columns]
    feature_df = joined_df[feature_columns]

    correlation_matrix = np.corrcoef(feature_df.T, goal_df.T)[:feature_df.shape[1], feature_df.shape[1]:]

    for i, feature_column in enumerate(feature_df.columns):
        for j, goal_column in enumerate(tables[code]):
            correlation = correlation_matrix[i, j]
            if not math.isnan(correlation):
                correlation_values[goal_column].append((correlation, feature_column))

    if save_file:
        with open(save_file, "w") as json_file:
            json.dump(correlation_values, json_file)

    return correlation_values


def visualise_osm_census_table_by_correlation(conn, type, code, size=16):
    tables = access.get_census_data_column_names()

    goal_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", tables[code] + ["OA"])
    feature_df = aws_utils.query_AWS_load_table(conn, type).drop(columns=["lat", "long"])

    goal_columns = tables[code]
    feature_columns = feature_df.columns.drop("OA")

    joined_df = goal_df.merge(feature_df, how="inner", on=["OA"])

    goal_df = joined_df[goal_columns]
    feature_df = joined_df[feature_columns]

    plot_utils.plot_correlation_heatmap_between_features(goal_df, feature_df, plot_size=(size, size / 2)) 


def visualise_osm_feture_against_density(conn, type, feature, size=10):
    columns = ["lat", "long", feature, "density"]
    census_df = aws_utils.query_AWS_load_table(conn, type, columns).drop(columns=["OA", "lat", "long"])

    plot_utils.plot_values_on_map_relative_to_median(census_df, base_figsize=(size/2, size/2), labels_on=False)
