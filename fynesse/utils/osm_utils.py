import osmium
import osmium.osm.mutable as osm_mutable
import pickle
import os
import numpy as np
import pandas as pd
from rtree import index



def filter_by_number_of_tags(input_file, output_file, min_num_tags=2):
    if os.path.exists(f"{output_file}"):
        print(f"`{output_file}` already exists")
        return
    
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
    if os.path.exists(f"{input_file}"):
        print(f"`{output_file}` already exists")
        return
    
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
    if os.path.exists(f"{input_file}"):
        print(f"`{output_file}` already exists")
        return
    
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
