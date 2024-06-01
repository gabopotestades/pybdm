import numpy as np
import random
from pybdm import BDM
from pybdm.partitions import PartitionCorrelated, PartitionIgnore, PartitionRecursive, PartitionPeriodic
from pybdm.algorithms import PerturbationExperiment
from community import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as ln
import netgraph as ng
import re
import os
import csv

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)    

def create_complete_graph(num_nodes=4):
    adj_matrix = np.ones((num_nodes, num_nodes), dtype=int)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def create_random_graph(num_nodes=4, probability = 0.5):
    # np.random.seed(456)
    # adj_matrix = np.random.randint(0, 2, (num_nodes,num_nodes))
    # np.fill_diagonal(adj_matrix, 0)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    indices = np.triu_indices(adj_matrix.shape[0], k=1)
    for i in range(0, indices[0].shape[0] - 1):
        edge = 1 if random.random() <= probability else 0
        adj_matrix[indices[0][i], indices[1][i]] = edge
    np.fill_diagonal(adj_matrix, 0)
    return np.triu(adj_matrix) + np.tril(adj_matrix.T)

def combine_two_graphs(first_graph, second_graph, num_nodes=4, randomize=False):
    real_nodes = num_nodes*2
    total_nodes = real_nodes
    G = np.zeros((total_nodes, total_nodes), dtype=int)
    G[:num_nodes, :num_nodes] = np.logical_or(
        G[:num_nodes, :num_nodes],
        first_graph
    ).astype(int)
    G[num_nodes:real_nodes, num_nodes:real_nodes] = np.logical_or(
        G[num_nodes:real_nodes, num_nodes:real_nodes],
        second_graph
    ).astype(int)
    G[0, -1] = 1
    G[-1, 0] = 1

    if not randomize:
        return G

    new_arrange = list(range(real_nodes))
    random.shuffle(new_arrange)
    return G[new_arrange][:, new_arrange]

def get_prefix(node):
    return re.search(r'(.*-)\d+', node).group(1)

def merge_graphs(graphs, rename, randomize=False, random_edges=None):
    
    rename_labels = rename
    edges_to_add = []
    added_edge_prefix = 'CN-'
    graphs_to_merge = graphs

    if random_edges is None:
        # connecting_node = nx.Graph()
        # connecting_node.add_node(1)
        # graphs_to_merge = [connecting_node]
        # graphs_to_merge.extend(graphs)
        rename_labels =  rename_labels + (added_edge_prefix,)

        for i in range(len(rename) - 1):
            edges_to_add.append((f'{rename[i]}0', f'{rename[i+1]}0'))
            
    else:
        # Get only first two
        first_graph_nodes = random.sample(list(graphs[0].nodes()), random_edges)
        second_graph_nodes = random.sample(list(graphs[1].nodes()), random_edges)

        for i in range(random_edges):
            edges_to_add.append((
                f'{rename[0]}{first_graph_nodes[i]}', f'{rename[1]}{second_graph_nodes[i]}'
            ))


    X = nx.union_all(graphs_to_merge, rename_labels)
    X.add_edges_from(edges_to_add)

    print(f'Number of Edges: {X.number_of_edges()}')
    print(f'Number of Nodes: {X.number_of_nodes()}')

    node_list = list(X.nodes)
    edge_list = [e for e in X.edges]
    edge_mapping = {v: i for i, v in enumerate(node_list)}

    prefix_color_mapping = {}
    color_mapping = {
        0 : 'tab:green',
        1 : 'tab:orange',
        2 : 'tab:red',
        3 : 'tab:blue',
        4 : 'tab:olive', 
        5 : 'tab:brown', 
        6 : 'tab:purple', 
        7 : 'tab:gray', 
        8 : 'tab:cyan', 
        9 : 'tab:pink'
    }
    for i, prefix in enumerate(rename_labels):
        prefix_color_mapping[prefix] = color_mapping[i % len(color_mapping)]

    edge_color = {}
    node_color = [prefix_color_mapping[get_prefix(node)] for node in X.nodes()]
    
    for edge in edge_list:
        u = edge[0]
        v = edge[1]
        if (get_prefix(u) == get_prefix(v)):
            e_color = prefix_color_mapping[get_prefix(u)].removeprefix('tab:')
        else:
            e_color = prefix_color_mapping[added_edge_prefix].removeprefix('tab:')

        u = edge_mapping[u]
        v = edge_mapping[v]
        edge_color[(u,v)] = e_color

    X = nx.to_numpy_array(X, dtype=int)

    if randomize:
        new_arrange = list(range(X.shape[0]))
        combined_list = list(zip(new_arrange, node_color))
        random.shuffle(combined_list)
        new_arrange, node_color = zip(*combined_list)
        new_arrange = list(new_arrange)
        new_arrange_mapping = {value: index for index, value in enumerate(new_arrange)}
        edge_color = {
            (new_arrange_mapping[k[0]], new_arrange_mapping[k[1]]): v for _, (k,v) in enumerate(edge_color.items())
        }
        node_color = list(node_color)
        return X[new_arrange][:, new_arrange], node_color, edge_color, prefix_color_mapping

    return X, node_color, edge_color, prefix_color_mapping

def convert_csv_to_graph(filename, randomize=False):

    reader = csv.reader(open(filename, 'r'), delimiter=',')
    original_graph = np.array(list(reader)).astype('int')
    added_edge_prefix = 'CN-'

    graphs = {
        'S-': nx.from_numpy_array(original_graph[:10,:10]),
        #'S-': nx.star_graph(21),
        'ER-': nx.from_numpy_array(original_graph[10:30, 10:30]),
        'C-': nx.from_numpy_array(original_graph[30:,30:]),
    }
    rename_labels = tuple(graphs.keys()) + (added_edge_prefix,)

    X = nx.union_all(graphs.values(), rename_labels)
    
    edges_to_add = [
        (f'{rename_labels[0]}0', f'{rename_labels[1]}11'),
        (f'{rename_labels[2]}0', f'{rename_labels[1]}11'),
    ]
    X.add_edges_from(edges_to_add)

    node_list = list(X.nodes)
    print(f'Node count: {len(node_list)}')
    edge_list = [e for e in X.edges]
    print(f'Edge count: {len(edge_list)}')
    edge_mapping = {v: i for i, v in enumerate(node_list)}

    prefix_color_mapping = {}
    color_mapping = {
        0 : 'tab:green',
        1 : 'tab:orange',
        2 : 'tab:red',
        3 : 'tab:blue',
        4 : 'tab:olive', 
        5 : 'tab:brown', 
        6 : 'tab:purple', 
        7 : 'tab:gray', 
        8 : 'tab:cyan', 
        9 : 'tab:pink'
    }
    for i, prefix in enumerate(rename_labels):
        prefix_color_mapping[prefix] = color_mapping[i % len(color_mapping)]

    edge_color = {}
    node_color = [prefix_color_mapping[get_prefix(node)] for node in X.nodes()]

    for edge in edge_list:
        u = edge[0]
        v = edge[1]
        if (get_prefix(u) == get_prefix(v)):
            e_color = prefix_color_mapping[get_prefix(u)].removeprefix('tab:')
        else:
            e_color = prefix_color_mapping[added_edge_prefix].removeprefix('tab:')

        u = edge_mapping[u]
        v = edge_mapping[v]
        edge_color[(u,v)] = e_color

    X = nx.to_numpy_array(X, dtype=int)
    # np.savetxt("res.csv", X, delimiter=",")

    if randomize:
        new_arrange = list(range(X.shape[0]))
        combined_list = list(zip(new_arrange, node_color))
        random.shuffle(combined_list)
        new_arrange, node_color = zip(*combined_list)
        new_arrange = list(new_arrange)
        new_arrange_mapping = {value: index for index, value in enumerate(new_arrange)}
        edge_color = {
            (new_arrange_mapping[k[0]], new_arrange_mapping[k[1]]): v for _, (k,v) in enumerate(edge_color.items())
        }
        node_color = list(node_color)
        return X[new_arrange][:, new_arrange], node_color, edge_color, prefix_color_mapping
    
    return X, node_color, edge_color, prefix_color_mapping

def draw_graph(filename, G, graphs_mapping, prefix_color_mapping, folder_name, node_color = None, colored_edges = np.empty((0, 2), dtype=int), edge_layout='curved', use_community=True,):

    with np.errstate(divide='ignore',invalid='ignore'):

        params = {}
        
        params['node_edge_width'] = 0
        params['node_labels'] = True
        params['node_layout']='community' if use_community else 'spring'
        params['edge_layout'] = edge_layout
        params['edge_layout_kwargs'] = dict(k=2000)
        edges_coords = [tuple(coord) for coord in colored_edges]
        params['edge_color'] = dict([
            ((u, v), '#ff0000') if (u, v) in edges_coords else ((u, v), '#2c404c') for u, v in G.edges()
        ])
        params['edge_alpha'] = dict([
            ((u, v), 0.5) if (u, v) in edges_coords else ((u, v), 0.15) for u, v in G.edges()
        ])

        if node_color:
            params['node_color'] = { node : color for node, color in enumerate(node_color) }

        if use_community:
            color_mapping = {
                0 : 'tab:orange',
                1 : 'tab:green',
                2 : 'tab:red',
                3 : 'tab:blue',
                4 : 'tab:purple', 
                5 : 'tab:brown', 
                6 : 'tab:pink', 
                7 : 'tab:gray', 
                8 : 'tab:olive', 
                9 : 'tab:cyan'
            }
            node_to_community = community_louvain.best_partition(G)
            params['node_layout_kwargs']=dict(node_to_community=node_to_community)

            if node_color is None:
                community_values = set(node_to_community.values())
                community_to_color = { c : color_mapping[c % len(color_mapping)] for c in community_values}
                community_node_color = {
                    node: community_to_color[community_id] for node, community_id in node_to_community.items()
                }
                params['node_color'] = community_node_color
        
        custom_lines = []
        legend_labels = []
        
        for graph_prefix in prefix_color_mapping:

            legend_labels.append(graphs_mapping[graph_prefix])
            custom_lines.append(
                ln.Line2D(
                    [0], [0], 
                    color=prefix_color_mapping[graph_prefix], 
                    marker = 'o',
                    linestyle='None',
                    markersize= 7
                )
            )

        plt.figure(dpi=200)
        ng.Graph(G, **params)

        plt.legend(
            custom_lines, 
            legend_labels,
            fontsize="7",
            loc="best"
        )
        # plt.title(f'Subgraphs = {nx.number_connected_components(G)}, Edges = {nx.number_of_edges(G)}')
        plt.title(f'{filename}')
        plt.savefig(f'{folder_name}/{filename}.png')
        plt.clf()

def draw_info_signature_plot(filename, info_loss, difference, difference_filter, auxiliary_cutoff, folder_name):
    info_loss_values = info_loss[:,-1]
    info_loss_yval = info_loss_values
    info_loss_xval = np.array([i for i in range(len(info_loss_values))])
    aux_cutoff_yval = np.array([auxiliary_cutoff + np.log2(2) for _ in range(len(info_loss_values))])

    edges_xval = np.array([i for i in range(len(difference))])
    marker_color = ['red' if x else 'blue' for x in difference_filter]

    plt.plot(info_loss_xval, info_loss_yval, 'orange', marker = 'o', markersize=3.5)
    plt.plot(edges_xval, difference, 'blue', linewidth=1)

    for i in range(len(difference)):
        plt.plot(edges_xval[i], difference[i], color=marker_color[i], marker='s', markersize=3.5, linewidth=0)
        
    plt.plot(info_loss_xval, aux_cutoff_yval, 'black', linewidth=1)
    
    custom_lines = [
        ln.Line2D([0], [0], color='orange', marker = 'o', lw=2),
        ln.Line2D([0], [0], color='blue', marker = 's', lw=2),
        ln.Line2D([0], [0], color='red', marker = 's', lw=2),
        ln.Line2D([0], [0], color='black', lw=2)
    ]

    plt.legend(
        custom_lines, 
        ['information signature', 'difference', 'deleted edges', 'log(2) + epsilon'],
        loc="upper right"
    )
    plt.title(filename)
    plt.xlabel("Sorted Edges")
    plt.ylabel("Information Value")
    plt.savefig(f'{folder_name}/{filename}.png')
    plt.clf()

def draw_edge_grouping_plot(filename, info_loss, edge_color, graphs_mapping, prefix_color_mapping, folder_name):
  
    coords = list(zip(info_loss[:,0].astype(int), info_loss[:,1].astype(int)))
    info_loss_yval = info_loss[:,-1]
    info_loss_xval = np.array([i for i in range(len(coords))])

    color_to_prefix = {v.removeprefix('tab:'): k for k, v in prefix_color_mapping.items()}
    marker_mapping = {
        'S-': 'D',
        'C-': '^',
        'ER-': 's',
        'CN-': 'o'
    }

    for i in range(len(coords)):

        mapped_color = edge_color[coords[i]] if coords[i] in edge_color else edge_color[coords[i][::-1]]

        params = {
            'color' : mapped_color,
            'marker' : marker_mapping[color_to_prefix[mapped_color]],
            'markersize': 3.5,
            'linewidth': 0
        }

        plt.plot(info_loss_xval[i], info_loss_yval[i], **params)
    
    custom_lines = []
    legend_labels = []
    
    for graph_prefix in prefix_color_mapping:

        legend_labels.append(graphs_mapping[graph_prefix])
        custom_lines.append(
            ln.Line2D(
                [0], [0], 
                color=prefix_color_mapping[graph_prefix], 
                marker = marker_mapping[graph_prefix],
                linestyle='None',
                markersize= 7
            )
        )

    plt.legend(
        custom_lines, 
        legend_labels,
        loc="upper right"
    )
    plt.title(filename)
    plt.xlabel("Sorted Edges")
    plt.ylabel("Information Value")
    plt.savefig(f'{folder_name}/{filename}.png')
    plt.clf()

nodes = 20

# ER = nx.to_numpy_array(nx.erdos_renyi_graph(nodes, 0.50))
# C = nx.to_numpy_array(nx.complete_graph(nodes))
# R = nx.to_numpy_array(nx.watts_strogatz_graph(nodes, 4, 0))
# BA = nx.to_numpy_array(nx.barabasi_albert_graph(nodes, nodes // 2))
# X = combine_two_graphs(ER, C, num_nodes=nodes, randomize=True)

# R = nx.watts_strogatz_graph(12, 4, 0)

graphs_mapping = {
    'S-': 'Star Graph',
    'C-': 'Complete Graph',
    'ER-': 'Erdos-Renyi Graph',
    'CN-': 'Connecting Edges'
}

# r1 = random.randint(0, 999999)
# print(f'Seed: {r1}')
# #289722

graphs = {
    'S-': nx.star_graph(21),
    'ER-': nx.gnm_random_graph(20, 20, 289722),
    'C-': nx.complete_graph(10),
}

# BA = nx.barabasi_albert_graph(100, 2)
# C = nx.complete_graph(20)
# X, node_color = merge_graphs([C, BA], rename=('C-', 'BA-'), randomize=True, random_edges=3)

# BA = nx.barabasi_albert_graph(100, 2)
# ER = nx.erdos_renyi_graph(nodes, 0.50)
# X, node_color = merge_graphs([ER, BA], rename=('ER-', 'BA-'), randomize=True, random_edges=3)

block_sizes = {
    #(3,3): '3x3',
    (4,4): '4x4',
}

partitions = [
    PartitionIgnore, 
    PartitionPeriodic,
    PartitionRecursive, 
    PartitionCorrelated
]

randomize_nodes = False

# S = nx.star_graph(17)
# C = nx.complete_graph(10)
# ER = nx.erdos_renyi_graph(13, 0.50)
# X, node_color, edge_color, prefix_color_mapping = merge_graphs(
#         list(graphs.values()), rename=tuple(graphs.keys()), randomize=randomize_nodes
#     )
X, node_color, edge_color, prefix_color_mapping = convert_csv_to_graph('testgraph.csv', False)

auxiliary_cutoff = 14
if True:

    for shape in block_sizes:

        print(f'=========={shape}==========')
        
        folder_name = 'results-' + block_sizes[shape]
        create_folder_if_not_exists(folder_name)
        submatrix_size = shape[0]

        for partition in partitions:

            print(f'\nProcessing {partition.__name__}:')

            params = {}
            has_sliding_window = partition == PartitionCorrelated
            shift_range = submatrix_size if has_sliding_window else 1

            for current_shift in range(1, shift_range + 1):

                file_name_prefix = f'Shift {current_shift} {partition.__name__} - ' if has_sliding_window else f'{partition.__name__} - '
                
                if has_sliding_window:
                    print(f'Current Sliding Window: {current_shift}') 
                    params['shift'] = current_shift

                original_graph = np.copy(X)
                perturbation = PerturbationExperiment(BDM(ndim=2, shape=shape, partition=partition, **params), X, metric='bdm')

                deco, info_loss, difference, difference_filter, deleted_edges, auxiliary_cutoff = perturbation.deconvolve_cutoff(auxiliary_cutoff=auxiliary_cutoff)

                # Checking
                # print(f'Original graph is equal: {np.array_equal(original_graph, perturbation.X)}')
                # print(f'Result is same with X: {np.array_equal(deco, perturbation.X)}')
                # print(f'Symmetrical: {check_symmetry(perturbation.X)}')

                # Setup graphs for drawing
                orig_graph = nx.from_numpy_array(original_graph)
                #deconvolve_graph = nx.from_numpy_array(deco)
                edge_layout = 'curved'
                use_community = True
                
                draw_graph(
                    f'{file_name_prefix}Deconvolution Graph', 
                    orig_graph, 
                    graphs_mapping,
                    prefix_color_mapping,
                    folder_name,
                    node_color=node_color, 
                    colored_edges=deleted_edges, 
                    edge_layout=edge_layout, 
                    use_community=use_community
                )
                # draw_graph(
                #     f'{file_name_prefix} Deconvoluted Graph', 
                #     deconvolve_graph, 
                #     node_color=node_color, 
                #     colored_edges=deleted_edges, 
                #     edge_layout=edge_layout, 
                #     use_community=use_community
                # )

                # Create Information Signature plot
                draw_info_signature_plot(
                    f'{file_name_prefix}Information Signature and Differences',
                    info_loss, 
                    difference, 
                    difference_filter, 
                    auxiliary_cutoff,
                    folder_name
                )

                # Create Edge Grouping Plot
                draw_edge_grouping_plot(
                    f'{file_name_prefix}Edge Grouping via Information Signature',
                    info_loss, 
                    edge_color,
                    graphs_mapping,
                    prefix_color_mapping,
                    folder_name
                )