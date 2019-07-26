"""
Generating graphs in graph edit distance classification problem.
"""
import numpy as np
import networkx as nx


def remove_edge(g, keep_connected=False, return_edge=False):
    edge_idx = np.random.choice(len(g.edges))
    edge_selected = list(g.edges)[edge_idx]

    if keep_connected:
        # might also have exception when there is only one edge between (u, v)
        edges = [edge for edge in g.edges if g.degree(edge[0]) > 1 and g.degree(edge[1]) > 1]
        edge_idx = np.random.choice(len(edges))
        edge_selected = edges[edge_idx]

    g.remove_edge(*edge_selected)

    if return_edge:
        return g, edge_selected
    return g


def add_edge_random(g, edge_duplicated=False, return_edge=False):
    # return_edge=True if need to return edge to check whether the add and remove edge are the same
    u, v = list(g.edges)[0]
    while (u, v) in g.edges:
        u = np.random.choice(g)
        ns = list(g)
        ns.remove(u)
        v = np.random.choice(ns)

    g.add_edge(u, v)

    if return_edge:
        return g, (u, v)
    return g


def change_one(g, changed=True, return_edges=True, print_change=True):
    g, edge_remove = remove_edge(g, return_edge=True)
    g, edge_add = add_edge_random(g, return_edge=True)
    if changed:
        while edge_add == edge_remove:
            print('equal edge: {} == {}'.format(edge_remove, edge_add))
            g.remove_edge(*edge_add)
            g, edge_add = add_edge_random(g, return_edge=True)

    print('remove_edge: {}, add_edge:{}'.format(edge_remove, edge_add))

    if return_edges:
        return g, edge_remove, edge_add

    return g


def generate_one_pair(n=20, p=0.2, connected=False, directed=False, triple=False, changed=True, print_change=True):
    """
    # change=True, the remove and add edge must be different

    if connected=True, the ER graph must be fully connected

    :param n:
    :param p:
    :param connected:
    :param directed:
    :param triple:
    :param changed:
    :return:
    """

    g1 = nx.erdos_renyi_graph(n, p, directed=directed)
    if connected:
        while nx.is_connected(g1) is False:
            g1 = nx.erdos_renyi_graph(n, p, directed=directed)

    g2, edge_remove2, edge_add2 = change_one(g1.copy(), changed=changed, return_edges=True, print_change=print_change)

    if triple:
        g3, edge_remove3, edge_add3 = change_one(g2.copy(), changed=changed, return_edges=True)
        while edge_remove3 == edge_add2 and edge_add3 == edge_remove2:  # in the case g1==g3
            g3, edge_remove3, edge_add3 = change_one(g2.copy(), changed=changed, return_edges=True)

        return g1, g2, g3

    else:
        return g1, g2


def generate_graph_pairs(n_sample=1000, n=20, p=0.2, connected=True, triple=False, directed=False, print_change=True):
    """
    generate graph pairs
    if triple=True: return list of <g1, g2, g3>
    else: return list of <g1, g2>

    :param n_sample:
    :param n:
    :param p:
    :param connected:
    :param triple:
    :param directed:
    :return:
    """
    g_list = []

    while len(g_list) < n_sample:
        g_pair = generate_one_pair(n=n, p=p, connected=connected, directed=directed, triple=triple, print_change=print_change)

        # sometime the network can be un-connected, for checking
        if all([nx.is_connected(g) for g in g_pair]):
            g_list.append(g_pair)

    return g_list


import torch

def get_tensors(g, input_size, max_edge=None):
    X = nx.to_numpy_matrix(g)
    if X.shape[0] <= input_size:
        input_size = 32
        pad_num = input_size - X.shape[0]
        N = X.shape[0]
        paddings = np.zeros(shape=[N, pad_num])
        X = np.concatenate([X, paddings], axis=-1)
    else:
        X = X[:, :input_size]

    X = np.ones([len(g), input_size])

    A = nx.to_numpy_matrix(g)
    A = torch.LongTensor(A)

    X = torch.FloatTensor(X)
    # E = torch.zeros([len(g.edges)*2, input_size])

    a = nx.to_numpy_matrix(g, dtype=int)
    e = np.ones([np.max(a), input_size])
    a[np.nonzero(a)] = np.cumsum(a[np.nonzero(a)], axis=-1)


    if len(e) < max_edge:
        pad_num = max_edge - len(e)
        paddings = np.zeros(shape=[pad_num, input_size])
        e = np.concatenate([e, paddings])

    E_index = torch.LongTensor(a)
    E = torch.FloatTensor(e)

    # E = torch.ones([len(g.edges) * 2, input_size])
    if max_edge:
        E = torch.ones([max_edge, input_size])
    return X, A, E_index, E


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    g_list = generate_graph_pairs(n_sample=5, n=20, p=0.2, connected=False, triple=False, directed=False, print_change=True)
    g1, g2 = g_list[0]
    pos = nx.spring_layout(g1)
    nx.draw(g1, with_labels=True, pos=pos)
    plt.show()

    nx.draw(g2, with_labels=True, pos=pos)
    plt.show()
