import osmium
import osmium.osm.mutable as osm_mutable
import pickle
import os
import numpy as np
import pandas as pd
from rtree import index



def filter_by_number_of_tags(input_file, output_file, min_num_tags=2):
    class NodeFilterHandlerFilter(osmium.SimpleHandler):
        def __init__(self):
            super(NodeFilterHandlerFilter, self).__init__()
            self.writer = osmium.SimpleWriter(output_file)

        def node(self, n):
            if len(n.tags) >= min_num_tags:
                self.writer.add_node(n)

        def close(self):
            self.writer.close()

    print("Filtering started")
    handler = NodeFilterHandlerFilter()
    handler.apply_file(input_file, locations=False)
    handler.close()
    print(f"Filtering complete. Output written to `{output_file}`\n")


def filter_and_save_selected_tags_only(input_file, output_file, selected_tags):
    class PublicTransportFilter(osmium.SimpleHandler):
        def __init__(self, output_file):
            super().__init__()
            self.writer = osmium.SimpleWriter(output_file)

        def node(self, n):
            for tag_key, tag_value in selected_tags:
                if n.tags.get(tag_key) == tag_value:
                    important_tags = {}
                    for tag_key, tag_value in n.tags:
                        if (tag_key, tag_value) in selected_tags:
                            important_tags[tag_key] = tag_value
                    self.writer.add_node(
                        osm_mutable.Node(
                            id=n.id,
                            location=n.location,
                            tags=important_tags
                        )
                    )

        def close(self):
            self.writer.close()

    print("Filtering started")
    handler = PublicTransportFilter(output_file)
    handler.apply_file(input_file, locations=True)
    handler.close()
    print(f"Filtering complete. Output written to `{output_file}`\n")


def filter_and_save_node(input_file, output_file, selected_tags):
    class PublicTransportFilter(osmium.SimpleHandler):
        def __init__(self, output_file):
            super().__init__()
            self.writer = osmium.SimpleWriter(output_file)

        def node(self, n):
            for tag_key, tag_value in selected_tags:
                if n.tags.get(tag_key) == tag_value:
                    self.writer.add_node(n)

        def close(self):
            self.writer.close()

    print("Filtering started")
    handler = PublicTransportFilter(output_file)
    handler.apply_file(input_file, locations=True)
    handler.close()
    print(f"Filtering complete. Output written to `{output_file}`\n")
    

def build_and_save_index(osm_file, index_file_name_base):
    if os.path.exists(f"{index_file_name_base}.pkl") or os.path.exists(f"{index_file_name_base}.dat") or os.path.exists(f"{index_file_name_base}.idx"):
        print(f"The index or the node files already exists, please choose another file name or delete them.")
        return

    class NodeHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.nodes = [] 

        def node(self, n):
            self.nodes.append((n.location.lat, n.location.lon, dict(n.tags)))

    print(f"Indexing OSM file `{osm_file}` into {index_file_name_base} (.pkl, .dat, .idx)")
    handler = NodeHandler()
    handler.apply_file(osm_file)
    nodes = handler.nodes

    with open(f"{index_file_name_base}.pkl", 'wb') as f:
        pickle.dump(nodes, f)

    idx = index.Index(index_file_name_base)
    for i, (lat, lon, _) in enumerate(nodes):
        idx.insert(i, (lon, lat, lon, lat))

    print(f"Indexing `{osm_file}` completed. \n")


def load_index_and_nodes(index_file_name_base):
    print(f"Loading data from {index_file_name_base}")
    if not os.path.exists(f"{index_file_name_base}.pkl") or not os.path.exists(f"{index_file_name_base}.dat") or not os.path.exists(f"{index_file_name_base}.idx"):
        print(f"The index or the node files are missing, please choose another source or create them.")
        return

    with open(f"{index_file_name_base}.pkl", 'rb') as f:
        nodes = pickle.load(f)

    idx = index.Index(index_file_name_base)
    return nodes, idx


def query_osm_from_index_files(nodes, idx, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    results = list(idx.intersection((lon_min, lat_min, lon_max, lat_max)))
    return np.array([nodes[i] for i in results])


def query_osm_in_batch(latitudes, longitudes, index_file_name_base, size_km=1, process_func=(lambda x : x)):
    # DO NOT EVER FORGET TO GIVE A process_func, OTHERWISE MEMORY WILL BE AN ISSUE
    delta = size_km / 222
    nodes, idx = load_index_and_nodes(index_file_name_base)
    results = []

    for lat, lon in zip(latitudes, longitudes):
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        results.append(process_func(query_osm_from_index_files(nodes, idx, bbox)))

    return results



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

public_transport_tags = [
    ("highway", "bus_stop"),
    ("railway", "station"),
    ("railway", "tram_stop")
]

# filter_by_number_of_tags("uk.osm.pbf", "uk_filtered_nodes.osm.pbf", min_num_tags=2)

# filter_and_save_selected_tags_only("uk_filtered_nodes.osm.pbf", "uk_amenities_non_transport.osm.pbf", amenity_non_transport_tags)
# filter_and_save_selected_tags_only("uk_filtered_nodes.osm.pbf", "uk_amenities_transport.osm.pbf", amenity_transport_tags)
# filter_and_save_node("uk_filtered_nodes.osm.pbf", "uk_public_transport_stops.osm.pbf", public_transport_tags)

# build_and_save_index("uk_amenities_non_transport.osm.pbf", "test_data/uk_amenities_non_transport")
# build_and_save_index("uk_amenities_transport.osm.pbf", "test_data/uk_amenities_transport")
# build_and_save_index("uk_public_transport_stops.osm.pbf", "test_data/uk_public_transport_stops")

lat, lon = 52.193433, 0.136374

latitudes, longitdues = np.full(5, lat), np.full(5, lon)
# latitudes += np.random.rand(100_000) / 10
# longitdues += np.random.rand(100_000) / 10

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
    

stops = query_osm_in_batch(latitudes, longitdues, "test_data/uk_public_transport_stops", process_func=stops_data_extractor, size_km=10)
stops_dicts = [transport_stops_record[0] for transport_stops_record in stops]
bus_stops_dicts = [transport_stops_record[1] for transport_stops_record in stops]
print(pd.concat([pd.DataFrame(stops_dicts), pd.DataFrame(bus_stops_dicts)], axis=1))


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


# amenity_count = query_osm_in_batch(latitudes, longitdues, "test_data/uk_amenities_non_transport", process_func=amenity_data_extractor, size_km=1)
# print(amenity_count)

