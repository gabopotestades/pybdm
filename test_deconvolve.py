import numpy as np
import random
from pybdm import BDM
from pybdm.partitions import PartitionRecursive
from pybdm.algorithms import PerturbationExperiment
from community import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as ln
import netgraph as ng
import re

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

def merge_graphs(graphs, rename, randomize=False, random_edges=None):

    def get_prefix(node):
        return re.search(r'(.*-)\d+', node).group(1)
    
    graphs_to_merge = graphs
    rename_labels = rename
    edges_to_add = []

    if random_edges is None:
        connecting_node = nx.Graph()
        connecting_node.add_node(1)
        graphs_to_merge.append(connecting_node)
        rename_labels = rename_labels + ('CN-',)

        for i in range(len(rename)):
            edges_to_add.append((f'{rename[i]}0', 'CN-1'))
            
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

    prefix_color_mapping = {}
    color_mapping = {
        0 : 'tab:blue',
        1 : 'tab:orange',
        2 : 'tab:green',
        3 : 'tab:red',
        4 : 'tab:purple', 
        5 : 'tab:brown', 
        6 : 'tab:pink', 
        7 : 'tab:gray', 
        8 : 'tab:olive', 
        9 : 'tab:cyan'
    }
    prefixes = set(get_prefix(node) for node in X.nodes())
    for i, prefix in enumerate(prefixes):
        prefix_color_mapping[prefix] = color_mapping[i % len(color_mapping)]
    node_color = [prefix_color_mapping[get_prefix(node)] for node in X.nodes()]

    X = nx.to_numpy_array(X, dtype=int)

    if randomize:
        new_arrange = list(range(X.shape[0]))
        combined_list = list(zip(new_arrange, node_color))
        random.shuffle(combined_list)
        new_arrange, node_color = zip(*combined_list)
        new_arrange = list(new_arrange)
        node_color = list(node_color)
        return X[new_arrange][:, new_arrange], node_color

    return X, node_color

def draw_graph(G, filename, node_color = None, colored_edges = np.empty((0, 2), dtype=int), edge_layout='curved', use_community=True):

    with np.errstate(divide='ignore',invalid='ignore'):

        kwargs = {}
        
        kwargs['node_edge_width'] = 0
        kwargs['node_labels'] = True
        kwargs['node_layout']='community' if use_community else 'spring'
        kwargs['edge_layout'] = edge_layout
        kwargs['edge_layout_kwargs'] = dict(k=2000)
        edges_coords = [tuple(coord) for coord in colored_edges]
        kwargs['edge_color'] = dict([
            ((u, v), '#ff0000') if (u, v) in edges_coords else ((u, v), '#2c404c') for u, v in G.edges()
        ])
        kwargs['edge_alpha'] = dict([
            ((u, v), 0.5) if (u, v) in edges_coords else ((u, v), 0.15) for u, v in G.edges()
        ])

        if node_color:
            kwargs['node_color'] = { node : color for node, color in enumerate(node_color) }

        if use_community:
            color_mapping = {
                0 : 'tab:blue',
                1 : 'tab:orange',
                2 : 'tab:green',
                3 : 'tab:red',
                4 : 'tab:purple', 
                5 : 'tab:brown', 
                6 : 'tab:pink', 
                7 : 'tab:gray', 
                8 : 'tab:olive', 
                9 : 'tab:cyan'
            }
            node_to_community = community_louvain.best_partition(G)
            community_values = set(node_to_community.values())
            community_to_color = { c : color_mapping[c % len(color_mapping)] for c in community_values}
            community_node_color = {
                node: community_to_color[community_id] for node, community_id in node_to_community.items()
            }
            kwargs['node_layout_kwargs']=dict(node_to_community=node_to_community)
            if node_color is None:
                kwargs['node_color'] = community_node_color
            

        plt.figure(dpi=200)
        ng.Graph(G, **kwargs)
        plt.title(f'Subgraphs = {nx.number_connected_components(G)}, Edges = {nx.number_of_edges(G)}')
        plt.savefig(filename)
        plt.clf()

def draw_info_signature_graph(info_loss, difference, difference_filter, auxiliary_cutoff):
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
    plt.title(f'Information Signature and Differences')
    plt.xlabel("Sorted Edges")
    plt.ylabel("Positive Information Value")
    plt.savefig('Information Signatures.png')
    plt.clf()

nodes = 20

# ER = nx.to_numpy_array(nx.erdos_renyi_graph(nodes, 0.50))
# C = nx.to_numpy_array(nx.complete_graph(nodes))
# R = nx.to_numpy_array(nx.watts_strogatz_graph(nodes, 4, 0))
# BA = nx.to_numpy_array(nx.barabasi_albert_graph(nodes, nodes // 2))
# X = combine_two_graphs(ER, C, num_nodes=nodes, randomize=True)

# R = nx.watts_strogatz_graph(12, 4, 0)

S = nx.star_graph(17)
C = nx.complete_graph(10)
ER = nx.erdos_renyi_graph(13, 0.50)
X, node_color = merge_graphs([S, C, ER], rename=('S-', 'C-', 'ER-'), randomize=True)

# BA = nx.barabasi_albert_graph(100, 2)
# C = nx.complete_graph(20)
# X, node_color = merge_graphs([C, BA], rename=('C-', 'BA-'), randomize=True, random_edges=3)

# BA = nx.barabasi_albert_graph(100, 2)
# ER = nx.erdos_renyi_graph(nodes, 0.50)
# X, node_color = merge_graphs([ER, BA], rename=('ER-', 'BA-'), randomize=True, random_edges=3)

if True:

    original_graph = np.copy(X)
    perturbation = PerturbationExperiment(BDM(ndim=2, partition=PartitionRecursive), X, metric='bdm',)

    # deco, info_loss, difference, deleted_edges = perturbation.deconvolve(2)
    deco, info_loss, difference, difference_filter, deleted_edges, auxiliary_cutoff = perturbation.deconvolve_cutoff()

    # Checking
    # print(f'Original graph is equal: {np.array_equal(original_graph, perturbation.X)}')
    # print(f'Result is same with X: {np.array_equal(deco, perturbation.X)}')
    # print(f'Symmetrical: {check_symmetry(perturbation.X)}')

    # Setup graphs for drawing
    orig_graph = nx.from_numpy_array(original_graph)
    deconvolve_graph = nx.from_numpy_array(deco)
    edge_layout = 'curved'
    use_community = True
    draw_graph(
        orig_graph, 
        'Original Graph.png', 
        node_color=node_color, 
        colored_edges=deleted_edges, 
        edge_layout=edge_layout, 
        use_community=use_community
    )
    draw_graph(
        deconvolve_graph, 
        'Deconvoluted Graph.png', 
        node_color=node_color, 
        colored_edges=deleted_edges, 
        edge_layout=edge_layout, 
        use_community=use_community
    )

    # Create Information Signature Graph
    draw_info_signature_graph(info_loss, difference, difference_filter, auxiliary_cutoff)