import numpy as np
import operator
import itertools
import math
from collections import defaultdict

def sub_decompositions(basic_decomposition):
    order = int(np.sum(basic_decomposition))
    decompositions = []
    variations = defaultdict(lambda: [])
    for curr_len in range(1, len(basic_decomposition)):
        for sum_rule in itertools.combinations_with_replacement(range(curr_len), order):
            i = 0
            sum_rule = np.array(sum_rule)
            curr_pows = np.array([np.sum(sum_rule == i) for i in range(curr_len)])
            curr_pows = curr_pows[curr_pows != 0]
            sorted_pow = tuple(np.sort(curr_pows))
            variations[sorted_pow].append(tuple(curr_pows))
            decompositions.append(sorted_pow)
    if len(decompositions) > 1:
        i = 0
        counts = np.zeros(len(variations))
        for dec, var in enumerate(variations):
            counts[i] = len(np.unique(var))
            i += 1
        decompositions = np.unique(decompositions)
    else:
        counts = np.ones(1)
    return decompositions, counts

def start_topo_sort(graph, visited, index, node, curr_index):
    index[node] = curr_index
    visited[node] = True
    curr_index += 1
    for child, _ in graph[node]:
        if not visited[node]:
            curr_index = start_topo_sort(graph, visited, index, child, curr_index)
            visited[node] = True
    return curr_index

def topo_sort(graph, node_list):
    num_nodes = len(graph.keys())
    visited = defaultdict(lambda: False)
    index = defaultdict(lambda: -1)
    curr_index = 0
    for node in node_list:
        if not visited[node]:
            curr_index = start_topo_sort(graph, visited, index, node, curr_index)
    return index

def local_coefficient(decomposition):
    order = np.sum(decomposition)
    coef = math.factorial(order)
    coef /= np.prod([math.factorial(x) for x in decomposition])
    _, counts = np.unique(decomposition, return_counts=True)
    coef /= np.prod([math.factorial(c) for c in counts])
    return coef

def powers_and_coefs(order):
    decompositions, _ = sub_decompositions(np.ones(order))
    graph = defaultdict(lambda: list())
    graph_reversed = defaultdict(lambda: list())
    for dec in decompositions:
        parents, weights = sub_decompositions(dec)
        for i in range(len(parents)):
            graph[parents[i]].append((dec, weights[i]))
            graph_reversed[dec].append((parents[i], weights[i]))

    topo_order = sorted(topo_sort(graph, decompositions).items(), key=operator.itemgetter(1))
    topo_order = [node for node, idx in topo_order]

    final_coefs = defaultdict(lambda: 0)
    for node in topo_order:
        local_coef = local_coefficient(node)
        final_coefs[node] += local_coef
        for p, w in graph_reversed[node]:
            final_coefs[p] -= w * final_coefs[node]
    powers_and_coefs = []
    for dec, c in final_coefs.iteritems():
        in_pows, out_pows = np.unique(dec, return_counts=True)
        powers_and_coefs.append((in_pows, out_pows, c))

    return powers_and_coefs
