from .config import *
import requests
import pymysql
import csv
import osmnx as ox
import warnings
import pandas as pd
import zipfile
import pickle
import io
from rtree import index
from shapely.geometry import box
from .utils import aws_utils, pandas_utils
from pyproj import Transformer
import osmium
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
        print (f"Downloading data for year: {year}")
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


def load_census_data(code, drop_culomns=None, column_names=None, ):
    try:
        census_df = load_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-oa.csv')

    except FileNotFoundError:
        census_df = load_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-msoa.csv')

    finally:
        if drop_culomns is not None:
            census_df = census_df.drop(census_df.columns[drop_culomns], axis=1)

        if column_names is not None:
            census_df.columns = column_names

        return census_df


# Project, Task 1


def clear_ONS_data_cords(source_file="Output_Areas_2021_PWC_V3_1988140134396269925.csv", destination_file="oa_cords.csv"):
    cords_df = pandas_utils.load_csv(source_file, ["OA21CD", "x", "y"])

    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    cords_df[['y', 'x']] = cords_df.apply(lambda row: pd.Series(transformer.transform(row['x'], row['y'])), axis=1)
    cords_df.columns = ["OA", "long", "lat"]
    cords_df.to_csv(destination_file, index=False)


def clear_ONS_data_hierarchy(source_file="Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv", destination_file="oa_hierarchy_mappings.csv"):
    hierarchy_df = pandas_utils.load_csv(source_file, ["OA21CD", "LSOA21CD", "LSOA21NM", "MSOA21CD", "MSOA21NM", "LAD22CD", "LAD22NM"])
    hierarchy_df.columns = ["OA", "LSOA", "LSOA_name", "MSOA", "MSOA_name", "LAD", "LAD_name"]
    
    hierarchy_df.to_csv(destination_file, index=False)



def upload_ONS_data(conn, base_dir="", 
                    source_file_names = ["Output_Areas_2021_PWC_V3_1988140134396269925.csv", "Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"],
                    destination_file_names=["oa_crds.csv", "oa_hierarchy_mapping.csv"], 
                    table_names=["oa_cords", "oa_hierarchy_mapping"], 
                    types=[["int(16)", "varchar(16)", "float(32)", "float(32)"], ["varchar(16)", "varchar(16)", "varchar(50)"]], 
                    keys=["OA", "OA"]):
    
    source_files = [os.path.join(base_dir, source_file_name) for source_file_name in source_file_names]
    destination_files = [os.path.join(base_dir, destination_file_name) for destination_file_name in destination_file_names]

    print("Clear ONS Coordinates Data")
    clear_ONS_data_cords(source_files[0], destination_files[0])
    print("Clear ONS Hierarchy Data")
    clear_ONS_data_hierarchy(source_files[1], destination_files[1])
    
    print("Uploading Cleared ONS Data To AWS")
    for destination_file, table_name, type, key in zip(destination_files, table_names, types, keys):
        aws_utils.upload_data_from_file(conn, destination_file, table_name, type, key)

    print("ONS Data Uploaded Successfully!")







def filter_osm_data_based_on_tags(input_file="uk.osm.pbf", output_file="uk_filtered.osm.pbf", min_tags=2):
    """
    Filter the osm data to exclude locations with too little tags.
    """
    class NodeFilterHandler(osmium.SimpleHandler):
        def __init__(self):
            super(NodeFilterHandler, self).__init__()
            self.writer = osmium.SimpleWriter(output_file)

        def node(self, n):
            if len(n.tags) >= min_tags:
                self.writer.add_node(n)

        def close(self):
            self.writer.close()

    print("Filtering...")
    
    handler = NodeFilterHandler()
    handler.apply_file(input_file)

    handler.close()

    print("Filtering complete. Output written to 'uk_super_filtered.osm.pbf'.")


def index_osm_data_on_location(input_file="uk_filtered.osm.pbf", nodes_file="nodes.pkl", index_file="rtree_index"):
    """
    Index the osm data on its coordinates: input_file -> nodes_file, rtree_index.
    """
    class NodeHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.nodes = [] 

        def node(self, n):
            self.nodes.append((n.location.lat, n.location.lon, dict(n.tags)))

    print("Parsing OSM file...")
    handler = NodeHandler()
    handler.apply_file(input_file)
    nodes = handler.nodes

    print(f"Saving {len(nodes)} nodes to {nodes_file}...")
    with open(nodes_file, 'wb') as f:
        pickle.dump(nodes, f)

    print("Building R-tree index...")
    idx = index.Index(index_file)
    for i, (lat, lon, tags) in enumerate(nodes):
        idx.insert(i, (lon, lat, lon, lat))

    print(f"Index saved to {index_file}.")
    return nodes, idx


def query_osm_batch(latitudes, longitudes, nodes_file="nodes.pkl", index_file="retree_index", tags=None, distance_km=1.0):
    """
    Query the OSM data in batch.
    """
    def load_index_and_nodes(nodes_file, index_file):
        """
        Load nodes and R-tree index from files.
        """
        with open(nodes_file, 'rb') as f:
            nodes = pickle.load(f)

        idx = index.Index(index_file)
        return nodes, idx
    
    nodes, idx = load_index_and_nodes(nodes_file, index_file)

    delta = distance_km / 111

    results = []
    for lat, lon in zip(latitudes, longitudes):
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)

        node_indices = list(idx.intersection(bbox))
        filtered_nodes = []

        for i in node_indices:
            node_lat, node_lon, node_tags = nodes[i]
            if tags is None or any(tag in node_tags for tag in tags):
                filtered_nodes.append((node_lat, node_lon, node_tags))

        results.append(filtered_nodes)

    return results

# Project, Task 2


# def add_key_to_table(conn, table_name, key):
#     cursor = conn.cursor()
    
#     sql_commands = f"""
#     ALTER TABLE `{table_name}`
#     ADD PRIMARY KEY (`{key}`);
#     """
    
#     for command in sql_commands.strip().split(';'):
#         if command.strip():
#             cursor.execute(command)
    
#     conn.commit()
#     print(f"Table `{table_name}` now has primary key `{key}`.")
#     cursor.close()


# def upload_csv_to_table(conn, table_name, file_name):
#     cur = conn.cursor()
#     cur.execute(f"LOAD DATA LOCAL INFILE '{file_name}' INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
#     conn.commit()


# def upload_census_data_from_df(conn, code, census_df, key = None, types=None):
#     table_name = "census_2021_" + code
#     if types is None:
#         types = ["float(32) NOT NULL" for _ in census_df.columns]
#         types[0] = "varchar(10) NOT NULL"

#     columns = "".join([f"`{field}` {t},\n" for field, t in zip(census_df.columns, types)])[:-2]

#     if key is None:
#         key = code + "_id"

#     setup_table(conn, table_name, columns)
#     add_key_to_table(conn, table_name, key)

#     census_df.to_csv("census_upload.csv", index=False)
#     upload_csv_to_table(conn, table_name, "census_upload.csv")


# def upload_ons_data_from_df(conn, ons_df, types=None):
#     table_name = "census_2021_oas"
#     if types is None:
#         types = ["float(32) NOT NULL" for _ in ons_df.columns]
#         types[0] = "varchar(10) NOT NULL"
#         types[1] = "varchar(10) NOT NULL"

#     columns = "".join([f"`{field}` {t},\n" for field, t in zip(ons_df.columns, types)])[:-2]

#     setup_table(conn, table_name, columns)
#     add_key_to_table(conn, table_name, "OA")

#     ons_df.to_csv("ons_upload.csv", index=False)
#     upload_csv_to_table(conn, table_name, "ons_upload.csv")


# def query_AWS_load_table(conn, table_name, columns=None):
#     if columns is None:
#         query_str = f"SELECT * FROM {table_name};"
#     else:
#         cols = ", ".join(columns)
#         query_str = f"SELECT {cols} FROM {table_name};"
    
#     cur = conn.cursor()
    
#     cur.execute(query_str)
#     data = cur.fetchall()
#     colnames = [desc[0] for desc in cur.description]
    
#     df = pd.DataFrame(data, columns=colnames)
#     return df


# def query_AWS_census_data(conn, code, column_names):
#     columns = column_names[code]
#     columns[0] = "OA"
#     return query_AWS_load_table(conn, "census_2021_joined", columns)
