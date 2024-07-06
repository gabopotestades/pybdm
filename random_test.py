import csv
import numpy as np
from collections import Counter
from itertools import product
import pandas as pd

# Function to generate indices
def ind(mat_dim, block_size, offset):
    return [range(i, j + 1) for i, j in zip(range(0, mat_dim - block_size + 1, offset), range(block_size - 1, mat_dim, offset))]

# Function to partition matrix
def my_partition(mat, block_size, offset):
    row_indices = ind(mat.shape[0], block_size, offset)
    print(row_indices)
    col_indices = ind(mat.shape[1], block_size, offset)
    print(col_indices)
    partitions = [mat[np.ix_(rows, cols)] for rows, cols in product(row_indices, col_indices)]
    print(partitions)
    return partitions

# Function to stringify a small block
def stringify(small_block):
    return ''.join(map(str, small_block.flatten()))


def count_py_implem():
    reader = csv.reader(open('testgraph.csv', 'r'), delimiter=',')
    original_graph = np.array(list(reader)).astype('int')
    string_list = []

    original_graph[[0,8], [8,0]] = 0

    for y in range(0, original_graph.shape[0], 4):
        for x in range(0, original_graph.shape[1], 4):
            sub_matrix = original_graph[y:y+4,x:x+4]
            flat_matrix = sub_matrix.flatten()
            matrix_string = ''.join([str(val) for val in flat_matrix])
            string_list.append(matrix_string)

    string_list.sort()

    counted = Counter(string_list)

    with open('py-1-9-test.csv','w') as csvfile:
        fieldnames=['flatSquares','lookup_values','frequency']
        writer=csv.writer(csvfile)
        writer.writerow(fieldnames)
        for key in counted:
            writer.writerow([key, '------------------', counted[key]]) 

def count_r_implem():
    reader = csv.reader(open('testgraph.csv', 'r'), delimiter=',')
    original_graph = np.array(list(reader)).astype('int')

    original_graph[[0,8], [8,0]] = 0

    parts = my_partition(original_graph, 4, 4)
    flat_squares = list(map(stringify, parts))

    squares_tally = pd.Series(flat_squares).value_counts().reset_index()
    squares_tally.columns = ['flatSquares', 'Freq']
    squares_tally.set_index('flatSquares', inplace=True)

    csv_data = pd.DataFrame({
        'flatSquares': squares_tally.index,
        'lookup_values':'------------------',
        'frequency': squares_tally['Freq'].values
    })

    # csv_data = csv_data.append({
    #     'flatSquares': 'bdm',
    #     'lookup_values': '------------------',
    #     'frequency': 0
    # }, ignore_index=True)

    csv_data = csv_data.sort_values(by=['flatSquares'])


    csv_data.to_csv('r-1-9-test.csv', index=False)

count_r_implem()