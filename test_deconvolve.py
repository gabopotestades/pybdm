import numpy as np
import random
from pybdm import BDM
from pybdm.algorithms import PerturbationExperiment
from networkx import from_numpy_array, draw, selfloop_edges, number_connected_components, number_of_edges
from matplotlib.pyplot import savefig, clf, title
from itertools import product

def check_symmetric(a, rtol=1e-05, atol=1e-08):
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

def combine_two_graphs(first_graph, second_graph, num_nodes=4):
    real_nodes = num_nodes*2
    total_nodes = real_nodes# + 1
    main_graph = np.zeros((total_nodes, total_nodes), dtype=int)
    main_graph[:num_nodes, :num_nodes] = np.logical_or(
        main_graph[:num_nodes, :num_nodes],
        first_graph
    ).astype(int)
    main_graph[num_nodes:real_nodes, num_nodes:real_nodes] = np.logical_or(
        main_graph[num_nodes:real_nodes, num_nodes:real_nodes],
        second_graph
    ).astype(int)
    main_graph[0, -1] = 1
    main_graph[real_nodes-1, -1] = 1
    return main_graph

nodes = 22
A = create_random_graph(nodes)
B = create_complete_graph(nodes)
X = combine_two_graphs(A, B, nodes)
#print(X)
# print(np.column_stack(np.nonzero(X)))
original_graph = np.copy(X)
perturbation = PerturbationExperiment(BDM(ndim=2), X, metric='bdm')
# orig_value = perturbation._value
# new_value = perturbation.run(idx=np.array([[0,1]]))
# print(orig_value)
# print(new_value)

#deco = perturbation.deconvolve(2)
deco = perturbation.deconvolve_cutoff()
# print(original_graph)
# print(perturbation.X)
# print(deco)

print(f'Original graph is equal: {np.array_equal(original_graph, perturbation.X)}')
print(f'Result is same with X: {np.array_equal(deco, perturbation.X)}')
print(f'Symmetrical: {check_symmetric(perturbation.X)}')

orig_graph = from_numpy_array(original_graph)
orig_graph.remove_edges_from(selfloop_edges(orig_graph))

deconvolve_graph = from_numpy_array(deco)
deconvolve_graph.remove_edges_from(selfloop_edges(deconvolve_graph))

title(f'Subgraphs = {number_connected_components(orig_graph)}, Edges = {number_of_edges(orig_graph)}')
draw(orig_graph, with_labels = True)
savefig("orig.png")
clf()
title(f'Subgraphs = {number_connected_components(deconvolve_graph)}, Edges = {number_of_edges(deconvolve_graph)}')
draw(deconvolve_graph, with_labels = True)
savefig("deco.png")

# A = np.array([
#     [0, 1, 1, 0],
#     [1, 0, 0, 1],
#     [1, 0, 0, 1],
#     [0, 1, 1, 0]], dtype=int)
# perturbation = PerturbationExperiment(BDM(ndim=2), A, metric='bdm')
# #idxer = np.array([[0,1], [0,2]], dtype=int)
# #valuer = np.array([0,0], dtype=int)

# valuer=np.full(A.shape[0]**2, 0, dtype=int)
# idxer = np.where(A > -1)
# idxer = np.array([idxer[0], idxer[1]]).T

# # tester = perturbation.run(idx=idxer, values=valuer)
# # print(tester)
# indexes = [ range(k) for k in A.shape ]
# idx = np.array([ x for x in product(*indexes) ], dtype=int)
# conncater=np.column_stack((idx, valuer))
# res = np.apply_along_axis(
#             lambda r: tuple(r[:-1]),
#             axis=1,
#             arr=conncater
#         )
# print(res)
# # test_graph = from_numpy_array(A)
# test_graph.remove_edges_from(selfloop_edges(test_graph))
# draw(test_graph, with_labels = True)
# savefig("test.png")


# B = np.arange(16).reshape(4,-1)
# query = (np.array([], dtype=int), np.array([], dtype=int))
# query2 = np.where(B%2 == 1)
# print(B[query2])
# res = (np.concatenate((query[0], query2[0])), np.concatenate((query[1], query2[1])))
# print(res[0])
# print(res[1])