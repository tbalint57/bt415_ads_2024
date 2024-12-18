from .config import *
import requests
import pymysql
import csv
import osmnx as ox
import warnings
import pandas as pd
import numpy as np
import zipfile
import io
from shapely.geometry import box
from .utils import aws_utils, pandas_utils, osm_utils
from pyproj import Transformer
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


def hello_world():
  print("Hello from the data science library!")


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print(f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
  """ Create a database connection to the MariaDB database
      specified by the host url and database name.
  :param user: username
  :param password: password
  :param host: host url
  :param database: database name
  :param port: port number
  :return: Connection object or None
  """
  conn = None
  try:
      conn = pymysql.connect(user=user,
                              passwd=password,
                              host=host,
                              port=port,
                              local_infile=1,
                              db=database
                              )
      print(f"Connection established!")
  except Exception as e:
      print(f"Error connecting to the MariaDB Server: {e}")
  return conn


def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))


def query_osm(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Access Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    # This might not be mathematically accurate, but can change it
    distance_coords = distance_km / 111
    north = latitude + distance_coords
    south = latitude - distance_coords
    west = longitude - distance_coords
    east = longitude + distance_coords

    bbox = box(west, south, east, north)
    pois = ox.features_from_polygon(bbox, tags=tags)

    return pois


def download_census_data(code, base_dir=''):
    url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Files extracted to: {extract_dir}")

    except:
        os.rmdir(extract_dir)
        print(f"{code} is not part of the dataset")


def load_csv(file_name, columns=None, column_names=None, index=None):
    df = pd.read_csv(file_name)

    if columns is None:
        return df

    df = df[columns]
    df.columns = column_names

    if index is None:
        return df

    df.set_index(index)
    return df


def load_census_data(code, base_dir=".", partition="oa", drop_culomns=None, column_names=None):
    try:
        census_df = load_csv(f'{base_dir}/census2021-{code.lower()}/census2021-{code.lower()}-{partition}.csv')
        
        if drop_culomns is not None:
            census_df = census_df.drop(census_df.columns[drop_culomns], axis=1)

        if column_names is not None:
            census_df.columns = column_names

        return census_df

    except FileNotFoundError:
        return None
    

def load_census_data_smallest_partition(code, base_dir=".", drop_culomns=None, column_names=None, partitions = ["oa", "lsoa", "msoa", "ltla", "utla", "rgn", "ctry"]):
    census_df = None

    for partition in partitions:
        census_df = load_census_data(code, base_dir=base_dir, partition=partition, drop_culomns=drop_culomns, column_names=column_names)
        if census_df is not None:
            return partition.upper(), census_df
        
    return None, None


# ONS


def clear_ONS_data_cords(source_file="Output_Areas_2021_PWC_V3_1988140134396269925.csv", destination_file="oa_cords.csv"):
    cords_df = pandas_utils.load_csv(source_file, ["OA21CD", "x", "y"])

    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    cords_df["long"], cords_df["lat"] = transformer.transform(cords_df["x"].values, cords_df["y"].values)
    cords_df = cords_df.rename(columns={"OA21CD": "OA"})[["OA", "lat", "long"]]
    cords_df.to_csv(destination_file, index=False)


def clear_ONS_data_hierarchy(source_file="Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv", destination_file="oa_hierarchy_mappings.csv"):
    hierarchy_df = pandas_utils.load_csv(source_file, ["OA21CD", "LSOA21CD", "LSOA21NM", "MSOA21CD", "MSOA21NM", "LAD22CD", "LAD22NM"])
    hierarchy_df = hierarchy_df.rename(columns={"OA21CD": "OA", 
                                        "LSOA21CD": "LSOA", 
                                        "LSOA21NM": "LSOA_name",
                                        "MSOA21CD": "MSOA", 
                                        "MSOA21NM": "MSOA_name", 
                                        "LAD22CD": "LAD", 
                                        "LAD22NM": "LAD_name"})
    
    print(hierarchy_df.columns)
    hierarchy_df.to_csv(destination_file, index=False)


def upload_ONS_data(conn, 
                    base_dir=".", 
                    source_file_names=["Output_Areas_2021_PWC_V3_1988140134396269925.csv", "Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"],
                    destination_file_names=["oa_crds.csv", "oa_hierarchy_mapping.csv"], 
                    table_names=["oa_cords", "oa_hierarchy_mapping"], 
                    types=[["varchar(16)", "float(32)", "float(32)"], ["varchar(16)", "varchar(16)", "varchar(50)", "varchar(16)", "varchar(50)", "varchar(16)", "varchar(50)"]], 
                    keys=["OA", "OA"]):
    
    source_files = [base_dir + "/" + source_file_name for source_file_name in source_file_names]
    destination_files = [base_dir + "/" + destination_file_name for destination_file_name in destination_file_names]

    print("Clear ONS Coordinates Data")
    clear_ONS_data_cords(source_files[0], destination_files[0])
    print("Clear ONS Hierarchy Data")
    clear_ONS_data_hierarchy(source_files[1], destination_files[1])
    
    print("Uploading Cleared ONS Data To AWS")
    for destination_file, table_name, type, key in zip(destination_files, table_names, types, keys):
        aws_utils.upload_data_from_file(conn, destination_file, table_name, type, key, indexed_columns=[])

    print("ONS Data Uploaded Successfully!")


# Cesnus


def calculate_close_points(lats, longs, distance_km=0.5):
    distance_cords = distance_km / 111

    points_2d = np.column_stack((lats, longs))

    tree = cKDTree(points_2d)

    neighbors_count = tree.query_ball_point(points_2d, r=distance_cords, return_length=True) - 1

    return neighbors_count


def upload_census_data(conn, 
                    base_dir="census_data", 
                    columns_to_drop=None, 
                    column_names=None, 
                    oa_cords_table_name="oa_cords", 
                    oa_hierarchy_table_name="oa_hierarchy_mapping"):

    if columns_to_drop is None:
        columns_to_drop = {
        "TS001": [0, 1, 3],
        "TS002": [0, 1, 3, 5, 6, 9, 13, 14, 16, 17, 19, 20],
        "TS003": [0, 1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24],
        "TS004": [0, 1, 3, 4, 7, 8, 9, 10, 12],
        "TS007": [0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        "TS011": [0, 1, 3],
        "TS016": [0, 1, 3],
        "TS017": [0, 1, 3],
        "TS018": [0, 1, 3, 5],
        "TS019": [0, 1, 3],
        "TS021": [0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27],
        "TS025": [0, 1, 3],
        "TS029": [0, 1, 3, 6, 7, 8, 9],
        "TS030": [0, 1, 3],
        "TS037": [0, 1, 3],
        "TS038": [0, 1, 3, 4, 7],
        "TS039": [0, 1, 3, 6, 7, 9, 10],
        "TS040": [0, 1, 3],
        "TS058": [0, 1, 3],
        "TS059": [0, 1, 3, 4, 7],
        "TS060": [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45, 47, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 63, 64, 65, 66, 67, 68, 70, 71, 72, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "TS061": [0, 1, 3],
        "TS062": [0, 1, 3],
        "TS063": [0, 1, 3],
        "TS065": [0, 1, 3],
        "TS066": [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32],
        "TS067": [0, 1, 3],
        "TS068": [0, 1, 3],
        "TS077": [0, 1, 3],
        "TS078": [0, 1, 3],
    }

    if column_names is None:
        column_names = {
        "TS001": ["id", "household", "communal"],
        "TS002": ["id", "never_married", "married_opposite_sex", "married_same_sex", "civil_partnership_opposite_sex", "civil_partnership_same_sex", "separated", "divorced", "widowed"],
        "TS003": ["id", "one_person", "single_family", "other"],
        "TS004": ["id", "UK", "EU", "Europe_non_EU", "Africa", "Asia", "Americas", "Australia_Oceania_Antarctica", "British_Overseas"],
        "TS007": ["id", "4_minus", "5_to_9", "10_to_15", "16_to_19", "20_to_24", "25_to_34", "35_to_49", "50_to_64", "65_to_74", "75_to_84", "85_plus"],
        "TS011": ["id", "0", "1", "2", "3", "4"],
        "TS016": ["id", "born_in_UK", "10_plus", "5_to_10", "2_to_5", "5_minus"],
        "TS017": ["id", "0", "1", "2", "3", "4", "5", "6", "7", "8_plus"],
        "TS018": ["id", "born_in_UK", "4_minus", "5_to_7", "8_to_9", "10_to_14", "15", "16_to_17", "18_to_19", "20_to_24", "25_to_29", "30_to_44", "45_to_59", "60_to_64", "65_to_74", "75_to_84", "85_to_89", "90_plus"],
        "TS019": ["id", "enumeration_address", "student_address", "UK_address", "non_UK_address"],
        "TS021": ["id", "asian", "black", "mixed", "white", "other"],
        "TS025": ["id", "all_english_or_welsh", "some_adult_english_or_welsh", "some_child_english_or_welsh", "no_english_or_welsh"],
        "TS029": ["id", "main_language_english", "main_language_not_english"],
        "TS030": ["id", "no_religion", "christian", "buddhist", "hindu", "jewish", "muslim", "sikh", "other", "not_answered"],
        "TS037": ["id", "very_good", "good", "fair", "bad", "very_bad"],
        "TS038": ["id", "limited_a_lot", "limited_a_little", "long_term_codition", "healthy"],
        "TS039": ["id", "none", "19_minus", "20_to_49", "50_plus"],
        "TS040": ["id", "none", "1", "2_plus"],
        "TS058": ["id", "2_minus", "2_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_60", "60_plus", "home_office", "no_fixed_location"],
        "TS059": ["id", "15_minus", "16_to_30", "31_to_48", "49_plus"],
        "TS060": ["id", "A_agriculture", "B_mining", "C_manufacturing", "D_electricity", "E_water", "F_construction", "G_retail", "H_transport", "I_accommodation", "J_information", "K_finance", "L_real_estate", "M_scientific", "N_administrative", "O_public_administration", "P_education", "Q_human_social", "Other"],
        "TS061": ["id", "working_from_home", "underground_tram", "train", "bus", "taxi", "motorcycle", "car_driving", "car_passenger", "bicycle", "walk", "other"],
        "TS062": ["id", "L1_3", "L4_6", "L7", "L8_9", "L10_11", "L12", "L13", "L14", "L15"],
        "TS063": ["id", "manager", "professional", "technical", "administrative", "skilled", "caring", "sales", "operator", "elementary"],
        "TS065": ["id", "employed", "unemployed", "never_been_employed"],
        "TS066": ["id", "active_non_student", "active_student", "inactive", "other"],
        "TS067": ["id", "none", "level_1", "level_2", "apprentiticeship", "level_3", "level_4", "other"],
        "TS068": ["id", "student", "not_student"],
        "TS077": ["id", "heterosexual", "homosexual", "bisexual", "other", "no_answer"],
        "TS078": ["id", "same_as_sex", "no_specific_identity", "trans_woman", "trans_man", "other", "no_answer"]
    }

    codes = column_names.keys()

    for code in codes:
        column_names[code] = [code + "_" + column_name for column_name in column_names[code]]

    joined_df = aws_utils.query_AWS_load_table(conn, oa_cords_table_name)
    normalised_joined_df = joined_df.copy()

    joined_types = ["varchar(16)", "float(32)", "float(32)"]
    normalised_joined_types = ["varchar(16)", "float(32)", "float(32)"]

    oa_hierarchy = aws_utils.query_AWS_load_table(conn, oa_hierarchy_table_name)

    for code in codes:
        print(f"\nDownloading census data for code {code}")
        download_census_data(code, base_dir=base_dir)
        print(f"Cleaning up census data for code {code}")
        partition, census_df = load_census_data_smallest_partition(code, base_dir, columns_to_drop[code], column_names[code])

        if partition != "OA":
            census_df.rename(columns={census_df.columns[0]: partition}, inplace=True)
            census_df = pd.merge(oa_hierarchy[["OA", partition]], census_df, on=[partition], how="inner").drop([partition], axis=1)

        census_df.rename(columns={census_df.columns[0]: "OA"}, inplace=True)
        
        print(f"Merging census data for code {code}")
        joined_df = pd.merge(joined_df, census_df, on=["OA"], how="inner")

        normalised_census_df = pandas_utils.normalise_data_frame_by_rows(census_df, ["OA"])
        normalised_joined_df = pd.merge(normalised_joined_df, normalised_census_df, on=["OA"], how="inner")

    joined_df["density"] = calculate_close_points(joined_df["lat"], joined_df["long"])
    normalised_joined_df["density"] = joined_df["density"] / max(joined_df["density"])

    print(f"\nUploading census data")
    joined_types += ["int(32)" for _ in range(len(joined_df.columns) - 3)]
    normalised_joined_types += ["float(32)" for _ in range(len(joined_df.columns) - 3)]

    aws_utils.upload_data_from_df(conn, joined_df, "census_data", joined_types, "OA")
    aws_utils.upload_data_from_df(conn, normalised_joined_df, "normalised_census_data", normalised_joined_types, "OA")
    
    print("\nCensus Data Successfully Uploaded!")


def get_census_data_column_names(include_code=True):
    column_names = {
        "TS001": ["id", "household", "communal"],
        "TS002": ["id", "never_married", "married_opposite_sex", "married_same_sex", "civil_partnership_opposite_sex", "civil_partnership_same_sex", "separated", "divorced", "widowed"],
        "TS003": ["id", "one_person", "single_family", "other"],
        "TS004": ["id", "UK", "EU", "Europe_non_EU", "Africa", "Asia", "Americas", "Australia_Oceania_Antarctica", "British_Overseas"],
        "TS007": ["id", "4_minus", "5_to_9", "10_to_15", "16_to_19", "20_to_24", "25_to_34", "35_to_49", "50_to_64", "65_to_74", "75_to_84", "85_plus"],
        "TS011": ["id", "0", "1", "2", "3", "4"],
        "TS016": ["id", "born_in_UK", "10_plus", "5_to_10", "2_to_5", "5_minus"],
        "TS017": ["id", "0", "1", "2", "3", "4", "5", "6", "7", "8_plus"],
        "TS018": ["id", "born_in_UK", "4_minus", "5_to_7", "8_to_9", "10_to_14", "15", "16_to_17", "18_to_19", "20_to_24", "25_to_29", "30_to_44", "45_to_59", "60_to_64", "65_to_74", "75_to_84", "85_to_89", "90_plus"],
        "TS019": ["id", "enumeration_address", "student_address", "UK_address", "non_UK_address"],
        "TS021": ["id", "asian", "black", "mixed", "white", "other"],
        "TS025": ["id", "all_english_or_welsh", "some_adult_english_or_welsh", "some_child_english_or_welsh", "no_english_or_welsh"],
        "TS029": ["id", "main_language_english", "main_language_not_english"],
        "TS030": ["id", "no_religion", "christian", "buddhist", "hindu", "jewish", "muslim", "sikh", "other", "not_answered"],
        "TS037": ["id", "very_good", "good", "fair", "bad", "very_bad"],
        "TS038": ["id", "limited_a_lot", "limited_a_little", "long_term_codition", "healthy"],
        "TS039": ["id", "none", "19_minus", "20_to_49", "50_plus"],
        "TS040": ["id", "none", "1", "2_plus"],
        "TS058": ["id", "2_minus", "2_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_60", "60_plus", "home_office", "no_fixed_location"],
        "TS059": ["id", "15_minus", "16_to_30", "31_to_48", "49_plus"],
        "TS060": ["id", "A_agriculture", "B_mining", "C_manufacturing", "D_electricity", "E_water", "F_construction", "G_retail", "H_transport", "I_accommodation", "J_information", "K_finance", "L_real_estate", "M_scientific", "N_administrative", "O_public_administration", "P_education", "Q_human_social", "Other"],
        "TS061": ["id", "working_from_home", "underground_tram", "train", "bus", "taxi", "motorcycle", "car_driving", "car_passenger", "bicycle", "walk", "other"],
        "TS062": ["id", "L1_3", "L4_6", "L7", "L8_9", "L10_11", "L12", "L13", "L14", "L15"],
        "TS063": ["id", "manager", "professional", "technical", "administrative", "skilled", "caring", "sales", "operator", "elementary"],
        "TS065": ["id", "employed", "unemployed", "never_been_employed"],
        "TS066": ["id", "active_non_student", "active_student", "inactive", "other"],
        "TS067": ["id", "none", "level_1", "level_2", "apprentiticeship", "level_3", "level_4", "other"],
        "TS068": ["id", "student", "not_student"],
        "TS077": ["id", "heterosexual", "homosexual", "bisexual", "other", "no_answer"],
        "TS078": ["id", "same_as_sex", "no_specific_identity", "trans_woman", "trans_man", "other", "no_answer"]
    }

    codes = column_names.keys()

    if include_code:
        for code in codes:
            column_names[code] = [code + "_" + column_name for column_name in column_names[code][1:]]
    
    else:
        for code in codes:
            column_names[code] = column_names[code][1:]

    return column_names


def get_census_data_table_descriptions():
    titles = {
        "TS001": "Number of residents in households and communal establishments",
        "TS002": "Legal partnership status",
        "TS003": "Household composition",
        "TS004": "Country of birth",
        "TS007": "Age groups",
        "TS011": "Households by deprivation dimensions",
        "TS016": "Length of residence",
        "TS017": "Number of people in one household",
        "TS018": "Age of arrival in the UK",
        "TS019": "Migrant indicator",
        "TS021": "Ethnic group",
        "TS025": "Household language",
        "TS029": "Proficiency in English",
        "TS030": "Religion",
        "TS037": "General health",
        "TS038": "Disability",
        "TS039": "Provision of unpaid care in hours",
        "TS040": "Number of disabled people in the household",
        "TS058": "Distance travelled to work",
        "TS059": "Hours worked",
        "TS060": "Industry",
        "TS061": "Method used to travel to work",
        "TS062": "NS-SeC (National Statistics Socio-economic Classification)",
        "TS063": "Occupation",
        "TS065": "Employment history",
        "TS066": "Economic activity status",
        "TS067": "Highest level of qualification",
        "TS068": "Schoolchildren and full-time students",
        "TS077": "Sexual orientation",
        "TS078": "Gender identity"
    }


    attributes = get_census_data_column_names(include_code=False)

    
    table = [
        {"Code": code, "Title": titles[code], "Attributes": ", ".join(attrs)}
        for code, attrs in attributes.items()
    ]

    return pd.DataFrame(table)

#OSM


def upload_OSM_data(conn, source_file="uk.osm.pbf"):
    filtered_osm_file = "uk_nodes_with_tags.osm.pbf"
    osm_utils.filter_by_number_of_tags(source_file, filtered_osm_file)

    public_transport_stops_tags = [
        ("highway", "bus_stop"),
        ("railway", "station"),
        ("railway", "tram_stop")
    ]

    amenity_non_transport_tags=[
        ("amenity", "post_box"),
        ("amenity", "fast_food"),
        ("amenity", "cafe"),
        ("amenity", "restaurant"),
        ("amenity", "pub"),
        ("amenity", "atm"),
        ("amenity", "post_office"),
        ("amenity", "pharmacy"),
        ("amenity", "place_of_worship"),
        ("amenity", "bar"),
        ("amenity", "bank"),
        ("amenity", "dentist"),
        ("amenity", "social_facility"),
        ("amenity", "doctors"),
        ("amenity", "kindergarten"),
        ("amenity", "parcel_locker"),
        ("amenity", "library"),
        ("amenity", "veterinary"),
        ("amenity", "clinic"),
        ("amenity", "childcare"),
        ("amenity", "school"),
        ("amenity", "nightclub"),
        ("amenity", "theatre"),
        ("amenity", "police"),
        ("amenity", "cinema"),
        ("amenity", "marketplace"),
        ("amenity", "college"),
        ("amenity", "hospital"),
        ("amenity", "university"),
        ("amenity", "social_club"),
        ("amenity", "courthouse")
    ]

    amenity_transport_tags = [
        ("amenity", "bicycle_parking"),
        ("amenity", "parking"),
        ("amenity", "charging_station"),
        ("amenity", "parking_space"),
        ("amenity", "fuel"),
        ("amenity", "bicycle_rental"),
        ("amenity", "motorcycle_parking"),
        ("amenity", "taxi"),
        ("amenity", "car_wash"),
        ("amenity", "ferry_terminal"),
        ("amenity", "car_rental"),
        ("amenity", "bicycle_repair_station"),
        ("amenity", "bus_station"),
        ("amenity", "trolley_bay"),
        ("amenity", "driving_school")
    ]

    public_transport_stops_file = "uk_public_transport_stops.osm.pbf"
    amenity_non_transport_file = "uk_amenities_non_transport.osm.pbf"
    amenity_transport_file = "uk_amenities_transport.osm.pbf"

    osm_utils.filter_and_save_node(filtered_osm_file, public_transport_stops_file, public_transport_stops_tags)
    osm_utils.filter_and_save_selected_tags_only(filtered_osm_file, amenity_non_transport_file, amenity_non_transport_tags)
    osm_utils.filter_and_save_selected_tags_only(filtered_osm_file, amenity_transport_file, amenity_transport_tags)

    
    public_transport_stops_index_file = "uk_public_transport_stops.osm.pbf"
    amenity_non_transport_index_file = "uk_amenities_non_transport.osm.pbf"
    amenity_transport_index_file = "uk_amenities_transport.osm.pbf"

    osm_utils.build_and_save_index(public_transport_stops_file, public_transport_stops_index_file)
    osm_utils.build_and_save_index(amenity_non_transport_file, amenity_non_transport_index_file)
    osm_utils.build_and_save_index(amenity_transport_file, amenity_transport_index_file)

    oa_cords_table_name = "oa_cords"

    oa_cords_df = aws_utils.query_AWS_load_table(conn, oa_cords_table_name)

    latitudes = oa_cords_df["lat"].to_numpy()
    longitudes = oa_cords_df["long"].to_numpy()

    def stops_data_extractor(nodes):
        stops = {
            "bus": 0,
            "train": 0,
            "underground": 0,
            "tram": 0 
        }

        bus_stop_count_by_type = {
            "shelter": 0,
            "covered": 0,
            "lit": 0,
            "bench": 0,
            "wheelchair": 0,
            "passenger_information_display": 0
        }

        for node in nodes:
            node_tags = node[2]

            if node_tags.get("highway") == "bus_stop":
                stops["bus"] += 1
                for type in bus_stop_count_by_type:
                    if node_tags.get(type) == "yes":
                        bus_stop_count_by_type[type] += 1
            
            if node_tags.get("railway") == "station":
                if node_tags.get("station") == "subway":
                    stops["underground"] += 1
        
                elif node_tags.get("station") == "subway":
                    stops["underground"] += 1
                
                else:
                    stops["train"] += 1


        return stops, bus_stop_count_by_type


    def amenity_data_extractor(nodes):
        count = {}

        for node in nodes:
            node_tags = node[2]
            for tag_key, tag_value in node_tags.items():
                # We can do this as by assumption tag_key=amenity
                if tag_value not in count:
                    count[tag_value] = 0

                count[tag_value] += 1

        return count

    nearby_public_transport_stops = osm_utils.query_osm_in_batch(latitudes, longitudes, public_transport_stops_index_file, process_func=stops_data_extractor)
    stops_dicts = [transport_stops_record[0] for transport_stops_record in nearby_public_transport_stops]
    bus_stops_dicts = [transport_stops_record[1] for transport_stops_record in nearby_public_transport_stops]
    nearby_public_transport_stops_df = pd.concat([oa_cords_df, pd.DataFrame(stops_dicts), pd.DataFrame(bus_stops_dicts)], axis=1)
    aws_utils.upload_data_from_df(conn, nearby_public_transport_stops_df ,"nearby_stops", ["varchar(16)", "float(16)", "float(16)"] + ["int(16)" for _ in range(len(nearby_public_transport_stops_df.columns) - 3)], "OA")

    nearby_amenity_non_transport = osm_utils.query_osm_in_batch(latitudes, longitudes, amenity_non_transport_index_file, process_func=amenity_data_extractor)
    nearby_amenity_non_transport_df = pd.concat([oa_cords_df, pd.DataFrame(nearby_amenity_non_transport).fillna(0).astype(int)], axis=1)
    aws_utils.upload_data_from_df(conn, nearby_amenity_non_transport_df ,"nearby_amenity_non_transport", ["varchar(16)", "float(16)", "float(16)"] + ["int(16)" for _ in range(len(nearby_amenity_non_transport_df.columns) - 3)], "OA")

    nearby_amenity_transport = osm_utils.query_osm_in_batch(latitudes, longitudes, amenity_transport_index_file, process_func=amenity_data_extractor)
    nearby_amenity_transport_df = pd.concat([oa_cords_df, pd.DataFrame(nearby_amenity_transport).fillna(0).astype(int)], axis=1)
    aws_utils.upload_data_from_df(conn, nearby_amenity_transport_df ,"nearby_amenity_transport", ["varchar(16)", "float(16)", "float(16)"] + ["int(16)" for _ in range(len(nearby_amenity_transport_df.columns) - 3)], "OA")
