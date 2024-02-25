import numpy as np
import random
from pybdm import BDM
from pybdm.algorithms import PerturbationExperiment
from community import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
import netgraph as ng

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

    new_arrange = list(range(real_nodes))
    random.shuffle(new_arrange)
    return G[new_arrange][:, new_arrange]

def draw_graph(G, colored_edges, filename, edge_layout='curved'):
    node_to_community = community_louvain.best_partition(G)
    community_values = set(node_to_community.values())
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
    community_to_color = { c : color_mapping[c % 9] for c in community_values}
    node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}
    edges_coords = [tuple(coord) for coord in colored_edges]
    edge_colors = dict([((u, v), '#ff0000') if (u, v) in edges_coords else ((u, v), '#2c404c') for u, v in G.edges()])

    plt.figure(dpi=200)
    ng.Graph(G,
        node_color=node_color, node_edge_width=0, edge_alpha=0.1, node_labels = True,
        node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout=edge_layout, edge_layout_kwargs=dict(k=2000), edge_color=edge_colors,
    )
    plt.title(f'Subgraphs = {nx.number_connected_components(G)}, Edges = {nx.number_of_edges(G)}')
    plt.savefig(filename)
    plt.clf()

def draw_info_signature_graph(info_loss, difference, auxiliary_cutoff):
    print(f'Info Loss Size: {info_loss[:,-1].shape}')
    info_loss_xval = np.array([i for i in range(len(info_loss[:,-1]))])
    info_loss_yval = info_loss[:,-1]

    edges_xval = np.array([i for i in range(len(difference))])
    difference_values = difference
    aux_cutoff_yval = np.array([auxiliary_cutoff + np.log2(2) for _ in range(len(difference))])

    plt.plot(info_loss_xval, info_loss_yval, 'red', marker = 'o')
    plt.plot(edges_xval, difference_values, 'blue', marker = 's')
    plt.plot(edges_xval, aux_cutoff_yval, 'orange', linestyle='dotted')
    plt.legend(['information signature', 'difference', 'log(2) + epsilon'], loc="upper right")
    plt.title(f'Information Signature and Differences')
    plt.savefig('Information Signatures.png')
    plt.clf()

nodes = 16
A = create_random_graph(nodes)
B = create_complete_graph(nodes)
X = combine_two_graphs(A, B, num_nodes=nodes)

original_graph = np.copy(X)
perturbation = PerturbationExperiment(BDM(ndim=2), X, metric='bdm')
auxiliary_cutoff = 1

#deco = perturbation.deconvolve(2)
deco, info_loss, difference, deleted_edges = perturbation.deconvolve_cutoff(auxiliary_cutoff=auxiliary_cutoff)

# Checking
print(f'Original graph is equal: {np.array_equal(original_graph, perturbation.X)}')
print(f'Result is same with X: {np.array_equal(deco, perturbation.X)}')
print(f'Symmetrical: {check_symmetry(perturbation.X)}')

# Setup graphs for drawing
orig_graph = nx.from_numpy_array(original_graph)
deconvolve_graph = nx.from_numpy_array(deco)
edge_layout = 'curved'
draw_graph(orig_graph, deleted_edges, 'Original Graph.png', edge_layout=edge_layout)
draw_graph(deconvolve_graph, deleted_edges, 'Deconvoluted Graph.png', edge_layout=edge_layout)

# Create Information Signature Graph
draw_info_signature_graph(info_loss, difference, auxiliary_cutoff)

# Color high info loss edges
# edges_coords = [tuple(coord) for coord in top_loss]
# edge_colors = ['red' if (u, v) in edges_coords else 'black' for u, v in orig_graph.edges()]

# # Draw Original Graph
# axis('off')
# pos = nx.spring_layout(orig_graph, k=0.15, iterations=30)
# orig_nodes = nx.draw_networkx_nodes(orig_graph, pos=pos)
# orig_nodes.set_edgecolor('w')
# nx.draw_networkx_edges(orig_graph, pos=pos, edge_color=edge_colors)
# nx.draw_networkx_labels(orig_graph, pos=pos, font_size=10)
# title(f'Subgraphs = {nx.number_connected_components(orig_graph)}, Edges = {nx.number_of_edges(orig_graph)}')
# savefig("Original Graph.png")

# clf()

# # Draw Deconvolved Graph
# axis('off')
# edge_colors = ['red' if (u, v) in edges_coords else 'black' for u, v in deconvolve_graph.edges()]
# pos = nx.spring_layout(deconvolve_graph, k=0.15, iterations=20)
# decon_nodes = nx.draw_networkx_nodes(deconvolve_graph, pos=pos)
# decon_nodes.set_edgecolor('w')
# nx.draw_networkx_edges(deconvolve_graph, pos=pos, edge_color=edge_colors)
# nx.draw_networkx_labels(deconvolve_graph, pos=pos, font_size=10)
# title(f'Subgraphs = {nx.number_connected_components(deconvolve_graph)}, Edges = {nx.number_of_edges(deconvolve_graph)}')
# savefig("Deconvoluted Graph.png")