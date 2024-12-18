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
def visualise_census_data_values(conn, code, size=5):
    columns = access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_increasing(census_df, title="Value Set of Trensport Data", plot_size=(size, size))


def visualise_census_data_distribution(conn, code, size=5):
    columns = access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_distribution(census_df, base_figsize=(size, size))


def visualise_census_by_distance_from_median_on_map(conn, code, size=5):
    columns = ["lat", "long"] + access.get_census_data_column_names()[code]
    census_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", columns)
    plot_utils.plot_values_on_map_relative_to_median(census_df, base_figsize=(size, size))


def visualise_census_data_locally(conn, locations, code, size=3):
    columns = ["lat", "long"] + access.get_census_data_column_names()[code]
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


def visualise_census_by_clustering(conn, code, n_clusters=5, size=10):
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


# Functions to visualise the osm data
def visualise_osm_data_values(conn, columns=None, size=5):
    census_df = aws_utils.query_AWS_load_table(conn, "nearby_amenity_non_transport", columns)
    plot_utils.plot_values_increasing(census_df, title="Value Set of Trensport Data", plot_size=(size, size))


def visualise_osm_data_distribution(conn, columns=None, size=5):
    census_df = aws_utils.query_AWS_load_table(conn, "nearby_amenity_non_transport", columns)
    plot_utils.plot_values_distribution(census_df, base_figsize=(size, size))


def visualise_osm_by_distance_from_median_on_map(conn, columns=None, size=5):
    if columns is not None:
        columns = ["lat", "long"] + access.get_census_data_column_names()
    census_df = aws_utils.query_AWS_load_table(conn, "nearby_amenity_non_transport", columns)
    plot_utils.plot_values_on_map_relative_to_median(census_df, base_figsize=(size, size))


def visualise_osm_data_locally(conn, lat, lon, columns=None, size=5):
    if columns is not None:
        columns = ["lat", "long"] + access.get_census_data_column_names()
    census_df = aws_utils.query_AWS_load_table(conn, "nearby_amenity_non_transport", columns)
    plot_utils.plot_values_on_map_relative_to_median(census_df, loc=(lat, lon), base_figsize=(size, size))


# ----- ===== -----


# def visualise_relationship_for_field(features_df, field_name, goal_df, merge_on=["OA"]):
#     feature_df = features_df[merge_on + [field_name]]
#     df = pd.merge(feature_df, goal_df, on=merge_on)

#     for goal_col in goal_df.columns:
#         if goal_col in merge_on:
#             continue
            
#         a, b = np.polyfit(df[field_name], df[goal_col], 1)
#         plt.plot(df[field_name], a*df[field_name]+b, label=goal_col)

#     plt.xlabel(field_name)
#     plt.ylabel("Goal values")
#     plt.title("Relationship between " + field_name + " and the mode of transportation")

#     plt.legend() 
#     plt.show() 


# def calculate_relationship_for_field(features_df, field_name, goal_df, merge_on=["OA"]):
#     feature_df = features_df[merge_on + [field_name]]
#     df = pd.merge(feature_df, goal_df, on=merge_on)

#     avg, mx = 0, 0 

#     for goal_col in goal_df.columns:
#         if goal_col in merge_on:
#             continue
            
#         a, b = np.polyfit(df[field_name], df[goal_col], 1)
#         avg += a
#         mx = max(mx, abs(a))
    
#     avg /= len(goal_df.columns) - 1

#     return avg, mx


# def compare_single_fields(feature_df, goal_df, input_col, goal_col, merge_on=["OA"]):
#     df = pd.merge(feature_df, goal_df, on=merge_on)
    
#     plt.figure(figsize=(6, 4))
#     plt.scatter(df[input_col], df[goal_col], alpha=0.7)
#     plt.title(f'{input_col} vs {goal_col}')
#     plt.xlabel(input_col)
#     plt.ylabel(goal_col)
#     plt.grid(True)
#     plt.show()


# def visualise_transport_data(conn):
#     field_names = ["TS061_working_from_home", "TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
#     transport_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", field_names)
#     plt.figure(figsize=(18, 6))
#     plot_utils.visualise_feature_values_increasing(transport_df)
#     plt.show()


# def visualise_transport_data_outliers(conn, number_of_outiers=5000):
#     transport_field_names = ["TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
#     transport_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", transport_field_names)
#     plt.figure(figsize=(18, 6))
#     plot_utils.visualise_feature_outliers(transport_df, number_of_outiers)
#     plt.show()


# def compare_two_tables(conn, code_a, code_b):
#     column_names = {
#         "TS001": ["household", "communal"],
#         "TS002": ["never_married", "married_opposite_sex", "married_same_sex", "civil_partnership_opposite_sex", "civil_partnership_same_sex", "separated", "divorced", "widowed"],
#         "TS003": ["one_person", "single_family", "other"],
#         "TS004": ["UK", "EU", "Europe_non_EU", "Africa", "Asia", "Americas", "Australia_Oceania_Antarctica", "British_Overseas"],
#         "TS007": ["4_minus", "5_to_9", "10_to_15", "16_to_19", "20_to_24", "25_to_34", "35_to_49", "50_to_64", "65_to_74", "75_to_84", "85_plus"],
#         "TS011": ["0", "1", "2", "3", "4"],
#         "TS016": ["born_in_UK", "10_plus", "5_to_10", "2_to_5", "5_minus"],
#         "TS017": ["0", "1", "2", "3", "4", "5", "6", "7", "8_plus"],
#         "TS018": ["born_in_UK", "4_minus", "5_to_7", "8_to_9", "10_to_14", "15", "16_to_17", "18_to_19", "20_to_24", "25_to_29", "30_to_44", "45_to_59", "60_to_64", "65_to_74", "75_to_84", "85_to_89", "90_plus"],
#         "TS019": ["enumeration_address", "student_address", "UK_address", "non_UK_address"],
#         "TS021": ["asian", "black", "mixed", "white", "other"],
#         "TS025": ["all_english_or_welsh", "some_adult_english_or_welsh", "some_child_english_or_welsh", "no_english_or_welsh"],
#         "TS029": ["main_language_english", "main_language_not_english"],
#         "TS030": ["no_religion", "christian", "buddhist", "hindu", "jewish", "muslim", "sikh", "other", "not_answered"],
#         "TS037": ["very_good", "good", "fair", "bad", "very_bad"],
#         "TS038": ["limited_a_lot", "limited_a_little", "long_term_codition", "healthy"],
#         "TS039": ["none", "19_minus", "20_to_49", "50_plus"],
#         "TS040": ["none", "1", "2_plus"],
#         "TS058": ["2_minus", "2_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_60", "60_plus", "home_office", "no_fixed_location"],
#         "TS059": ["15_minus", "16_to_30", "31_to_48", "49_plus"],
#         "TS060": ["A_agriculture", "B_mining", "C_manufacturing", "D_electricity", "E_water", "F_construction", "G_retail", "H_transport", "I_accommodation", "J_information", "K_finance", "L_real_estate", "M_scientific", "N_administrative", "O_public_administration", "P_education", "Q_human_social", "Other"],
#         "TS061": ["working_from_home", "underground_tram", "train", "bus", "taxi", "motorcycle", "car_driving", "car_passenger", "bicycle", "walk", "other"],
#         "TS062": ["L1_3", "L4_6", "L7", "L8_9", "L10_11", "L12", "L13", "L14", "L15"],
#         "TS063": ["manager", "professional", "technical", "administrative", "skilled", "caring", "sales", "operator", "elementary"],
#         "TS065": ["employed", "unemployed", "never_been_employed"],
#         "TS066": ["active_non_student", "active_student", "inactive", "other"],
#         "TS067": ["none", "level_1", "level_2", "apprentiticeship", "level_3", "level_4", "other"],
#         "TS068": ["student", "not_student"],
#         "TS077": ["heterosexual", "homosexual", "bisexual", "other", "no_answer"],
#         "TS078": ["same_as_sex", "no_specific_identity", "trans_woman", "trans_man", "other", "no_answer"]
#     }
#     columns_a = [code_a + column_name for column_name in column_names[code_a]]
#     columns_b = [code_b + column_name for column_name in column_names[code_b]]
        
#     response_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", ["OA"] + columns_a + columns_b)

#     a_df = response_df[["OA"] + columns_a]
#     b_df = response_df[["OA"] + columns_b]

#     plt.figure(figsize=(9, 3 * len(columns_a)))
#     plot_utils.visualise_relationship_by_components(a_df, b_df)
#     plt.show()


def visualise_feature_on_map(conn, feature):
    feature_names = ["lat", "long", feature]
    feature_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", feature_names)

    plot_utils.plot_values_on_map_relative_to_median(feature_df, base_figsize=(9, 9))
    plt.show()


# def visualise_all_transport_usages_on_map(conn):
#     transport_field_names = ["lat", "long", "TS061_working_from_home", "TS061_underground_tram", "TS061_train", "TS061_bus", "TS061_taxi", "TS061_motorcycle", "TS061_car_driving", "TS061_car_passenger", "TS061_bicycle", "TS061_walk", "TS061_other"]
#     response_df = aws_utils.query_AWS_load_table(conn, "normalised_census_data", transport_field_names)

#     for feature_name in transport_field_names:
#         if feature_name in ["lat", "long"]:
#             continue

#         plt.figure(figsize=(12, 12))
#         plot_utils.visualise_feature_on_map_relative_to_median(response_df, feature_name)
#         plt.show()


# def visualise_relationship_between_two_fields(conn, field_a, field_b, table_name="normalised_census_data"):
#     response_df = aws_utils.query_AWS_load_table(conn, table_name, [field_b, field_a])

#     plt.figure(figsize=(9, 9))
#     plot_utils.visualise_relationship(response_df, field_a, field_b)
#     plt.show()
