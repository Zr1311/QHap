import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from functools import partial
import networkx as nx
import time

snp_data = None
row_mapping = None


def create_sparse_matrix_from_triple(matrix_triple):
    if not matrix_triple:
        return None

    row_indices = []
    col_indices = []
    values = []

    for triple in matrix_triple:
        if len(triple) != 3:
            continue
        row_idx, col_idx, value = triple
        if value in (-1, 0, 1):
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            values.append(value)

    if not row_indices:
        return None

    max_row = max(row_indices)
    max_col = max(col_indices)

    sparse_matrix = coo_matrix((values, (row_indices, col_indices)),
                               shape=(max_row + 1, max_col + 1),
                               dtype=np.int8)

    return sparse_matrix


def reverse_mapping(fragment_to_idx):
    idx_to_fragment = {}
    for fragment, idx in fragment_to_idx.items():
        idx_to_fragment[idx] = fragment
    return idx_to_fragment


def compute_column_spans(sparse_matrix):
    csc_mat = sparse_matrix.tocsc()
    num_cols = csc_mat.shape[1]
    spans = []

    for j in range(num_cols):
        col_indices = csc_mat.getcol(j).nonzero()[0]
        if col_indices.size > 0:
            start_row = col_indices[0]
            end_row = col_indices[-1]
            spans.append((start_row, end_row))
        else:
            spans.append((None, None))

    return spans


def initialize_globals_shared(data, mapping):
    global snp_data, row_mapping
    snp_data = data
    row_mapping = mapping


def process_column(column_index, spans):
    start_row, end_row = spans[column_index]
    if start_row is None:
        return {}

    col_data = snp_data.getcol(column_index).tocsr()
    non_zero_rows = col_data.nonzero()[0]
    values = col_data.data

    row_to_value = dict(zip(non_zero_rows, values))
    diff_counts = defaultdict(int)

    for row1 in range(start_row, end_row + 1):
        val1 = row_to_value.get(row1)
        if val1 is None:
            continue

        for row2 in range(row1 + 1, end_row + 1):
            val2 = row_to_value.get(row2)
            if val2 is None:
                continue

            if val1 != val2:
                pair = (row1, row2) if row1 < row2 else (row2, row1)
                diff_counts[pair] += 1

    return diff_counts


def create_snp_graph(sparse_data, mapping, spans):
    global snp_data, row_mapping
    snp_data = sparse_data.tocsc()
    row_mapping = mapping

    num_cols = snp_data.shape[1]
    column_indices = list(range(num_cols))
    pool_size = cpu_count()

    start_time = time.time()

    with Pool(processes=pool_size,
              initializer=initialize_globals_shared,
              initargs=(snp_data, row_mapping)) as pool:
        process_func = partial(process_column, spans=spans)
        results = pool.imap_unordered(process_func, column_indices, chunksize=100)

        edge_counts = defaultdict(int)
        for res in tqdm(results, total=num_cols, desc="Processing columns", unit="col"):
            for pair, count in res.items():
                edge_counts[pair] += count

    edges = []
    for (row1, row2), weight in edge_counts.items():
        if weight > 0:
            node1 = row_mapping.get(row1, str(row1))
            node2 = row_mapping.get(row2, str(row2))
            edges.append((node1, node2, weight))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Building SNP graph time: {elapsed_time:.2f} seconds")

    return edges


def build_graph(edges, all_nodes):
    G = nx.Graph()
    for node in all_nodes:
        G.add_node(str(node))
    for node1, node2, weight in edges:
        G.add_edge(str(node1), str(node2), weight=weight)
    return G


def find_connected_components(G):
    return list(nx.connected_components(G))


def create_component_graphs(G, components, edges):
    start_time = time.time()

    multi_node_components = [(i, comp) for i, comp in enumerate(components) if len(comp) > 1]
    single_node_count = len(components) - len(multi_node_components)

    if single_node_count > 0:
        print(f"Skipping {single_node_count} single-node components")

    node_to_component = {}
    new_component_index = {}

    for new_idx, (orig_idx, component) in enumerate(multi_node_components):
        new_component_index[orig_idx] = new_idx
        for node in component:
            node_to_component[node] = new_idx

    component_graphs = {}
    for new_idx, (orig_idx, component) in enumerate(multi_node_components):
        component_graphs[new_idx] = {
            'nodes': list(component),
            'edges': []
        }

    for node1, node2, weight in edges:
        node1_str = str(node1)
        node2_str = str(node2)
        comp_idx = node_to_component.get(node1_str)

        if comp_idx is not None:
            component_graphs[comp_idx]['edges'].append((node1, node2, weight))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Creating component subgraphs time: {elapsed_time:.2f} seconds")

    return component_graphs


def print_statistics(components, component_graphs):
    print("\n===== Statistics =====")
    single_node_components = sum(1 for comp in components if len(comp) == 1)
    multi_node_components = len(components) - single_node_components

    print(f"Total original components: {len(components)}")
    print(f"- Multi-node components: {multi_node_components}")
    print(f"- Single-node components: {single_node_components} (skipped)")
    print(f"Saved blocks: {len(component_graphs)}")

    total_nodes = sum(len(comp_info['nodes']) for comp_info in component_graphs.values())
    print(f"Total nodes in saved blocks: {total_nodes}")

    print("\nSaved block details:")
    for i, comp_info in component_graphs.items():
        print(f"Component {i}: {len(comp_info['nodes'])} nodes, {len(comp_info['edges'])} edges")


def main(matrix_triple, fragment_to_idx, all_nodes):
    print("===== Step 1: Create sparse matrix =====")
    sparse_data = create_sparse_matrix_from_triple(matrix_triple)

    if sparse_data is None or sparse_data.size == 0:
        return {}

    print("===== Step 2: Process mapping =====")
    idx_to_fragment = reverse_mapping(fragment_to_idx)

    if idx_to_fragment is None:
        return {}

    num_nodes = sparse_data.shape[0]
    print(f"Number of graph nodes: {num_nodes}")

    print("===== Step 3: Compute column spans =====")
    spans = compute_column_spans(sparse_data)

    print("===== Step 4: Build SNP graph =====")
    edges = create_snp_graph(sparse_data, idx_to_fragment, spans)
    print(f"Graph edges: {len(edges)}")

    print("===== Step 5: Build NetworkX graph =====")
    G = build_graph(edges, all_nodes)
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("===== Step 6: Find connected components =====")
    components = find_connected_components(G)
    print(f"Found {len(components)} connected components")

    print("===== Step 7: Create component subgraphs =====")
    component_graphs = create_component_graphs(G, components, edges)

    print_statistics(components, component_graphs)
    print("\n===== Processing completed =====")

    return component_graphs


def get_Snps_Map_reads_block(matrix_triple, row_mapping, all_nodes):
    return main(matrix_triple, row_mapping, all_nodes)