"""Supporting functions for arbitrary order Factorization Machines."""


import itertools
from itertools import combinations_with_replacement, takewhile, count
import math
from collections import defaultdict
import numpy as np
import tensorflow as tf

def get_shorter_decompositions(basic_decomposition):
    """Returns all arrays simpler than basic_decomposition.

    Returns all arrays that can be constructed from basic_decomposition
    via joining (summing) its elements.

    Parameters
    ----------
    basic_decomposition : list or np.array
        The array from which to build subsequent ones.

    Returns
    -------
    decompositions : list of tuples
        All possible arrays that can be constructed from basic_decomposition.
    counts : np.array
        counts[i] equals to the number of ways to build decompositions[i] from
        basic_decomposition.

    Example
    -------
    decompositions, counts = get_shorter_decompositions([1, 2, 3])
        decompositions == [(1, 5), (2, 4), (3, 3), (6,)]
        counts == [ 2.,  1.,  1.,  2.]
    """
    order = int(np.sum(basic_decomposition))
    decompositions = []
    variations = defaultdict(lambda: [])
    for curr_len in range(1, len(basic_decomposition)):
        for sum_rule in combinations_with_replacement(range(curr_len), order):
            i = 0
            sum_rule = np.array(sum_rule)
            curr_pows = np.array([np.sum(sum_rule == i) for i in range(curr_len)])
            curr_pows = curr_pows[curr_pows != 0]
            sorted_pow = tuple(np.sort(curr_pows))
            variations[sorted_pow].append(tuple(curr_pows))
            decompositions.append(sorted_pow)
    if len(decompositions) > 1:
        decompositions = np.unique(decompositions)
        counts = np.zeros(decompositions.shape[0])
        for i, dec in enumerate(decompositions):
            counts[i] = len(np.unique(variations[dec]))
    else:
        counts = np.ones(1)
    return decompositions, counts

def sort_topologically(children_by_node, node_list):
    """Topological sort of a graph.

    Parameters
    ----------
    children_by_node : dict
        Children for any node.
    node_list : list
        All nodes (some nodes may not have children and thus a separate
        parameter is needed).

    Returns
    -------
    list, nodes in the topological order
    """
    levels_by_node = {}
    nodes_by_level = defaultdict(set)

    def walk_depth_first(node):
        if node in levels_by_node:
            return levels_by_node[node]
        children = children_by_node[node]
        level = 0 if not children else (1 + max(walk_depth_first(lname) for lname, _ in children))
        levels_by_node[node] = level
        nodes_by_level[level].add(node)
        return level

    for node in node_list:
        walk_depth_first(node)

    nodes_by_level = list(takewhile(lambda x: x != [],
                                    (list(nodes_by_level[i]) for i in count())))
    return list(itertools.chain.from_iterable(nodes_by_level))

def initial_coefficient(decomposition):
    """Compute initial coefficient of the decomposition."""
    order = np.sum(decomposition)
    coef = math.factorial(order)
    coef /= np.prod([math.factorial(x) for x in decomposition])
    _, counts = np.unique(decomposition, return_counts=True)
    coef /= np.prod([math.factorial(c) for c in counts])
    return coef

def powers_and_coefs(order):
    """For a `order`-way FM returns the powers and their coefficients needed to
    compute model equation efficiently
    """
    decompositions, _ = get_shorter_decompositions(np.ones(order))
    graph = defaultdict(lambda: list())
    graph_reversed = defaultdict(lambda: list())
    for dec in decompositions:
        parents, weights = get_shorter_decompositions(dec)
        for i in range(len(parents)):
            graph[parents[i]].append((dec, weights[i]))
            graph_reversed[dec].append((parents[i], weights[i]))

    topo_order = sort_topologically(graph, decompositions)

    final_coefs = defaultdict(lambda: 0)
    for node in topo_order:
        final_coefs[node] += initial_coefficient(node)
        for p, w in graph_reversed[node]:
            final_coefs[p] -= w * final_coefs[node]
    powers_and_coefs_list = []
    # for dec, c in final_coefs.iteritems():
    for dec, c in final_coefs.items():
        in_pows, out_pows = np.unique(dec, return_counts=True)
        powers_and_coefs_list.append((in_pows, out_pows, c))

    return powers_and_coefs_list


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Predefined loss functions
# Should take 2 tf.Ops: outputs and targets and should return tf.Op of loss
# Be carefull about dimentionality -- maybe tf.transpose(outputs) is needed

def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)))
    return tf.minimum(raw_loss, 100, name='truncated_log_loss')


def loss_mse(outputs, y):
    return tf.pow(y -  tf.transpose(outputs), 2, name='mse_loss')

