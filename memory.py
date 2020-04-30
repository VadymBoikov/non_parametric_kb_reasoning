import numpy as np
import networkx as nx
import utils
from scipy.sparse import csr_matrix
from itertools import product
import pandas as pd


def get_relation_cases(edge, graph, cutoff, max_relations):
    # extract cases for one relation 
    u,v = edge
    path_generator = nx.all_simple_paths(graph, source=u, target=v, cutoff=cutoff)
    cases = set() # one node can have many equal relations
    for path in path_generator:
        relation_paths = []
        for i_step in range(len(path) - 1):
            # 2 nodes can have multiple relations
            edge_relations = graph.get_edge_data(path[i_step], path[i_step+1])['relations']
            relation_paths.append(edge_relations)

        for i in product(*relation_paths):  # all combination for relations in paths
            cases.add(tuple(i))
            if len(cases) >= max_relations:
                return cases

    return cases


@utils.timeit
def create_memory_cases(graph, cutoff, max_relations, cores=1):
    edges = graph.edges
    cases_list = utils.run_function_on_list(get_relation_cases, edges, graph=graph,
                                            cutoff=cutoff, max_relations=max_relations, cores=cores)
    cases = {}

    for i, row in enumerate(edges):
        u, v = row
        for relation in graph.get_edge_data(u,v)['relations']:
            name = (u, relation, v)
            cases[name] = cases_list[i]
    return cases


@utils.timeit
def create_similarity(G, sparse=False):
    # I recreate kb, because Graph is modified with inverse relations
    kb = []
    for (u,v) in G.edges:
        edge_relations = G.get_edge_data(u,v)['relations']
        for rel in edge_relations:
            kb.append([u, rel])
    kb = pd.DataFrame(kb, columns=['e1', 'r'])

    relations = sorted(np.unique(kb['r']))
    node_positions = {k: v for v,k in enumerate(sorted(np.unique(kb['e1'])))}
    relation_positions = {k: v for v,k in enumerate(relations)}

    if sparse:
        rows = np.array([node_positions[row.e1] for row in kb.itertuples()])
        cols = np.array([relation_positions[row.r] for row in kb.itertuples()])
        vals = np.ones(len(rows))
        node_embeddings = csr_matrix((vals, (rows, cols)),  shape = (len(node_positions), len(relations)), dtype=np.long)
        node_embeddings[node_embeddings > 1] = 1  # if entity have more than 1 same relation
    else:
        node_embeddings = np.zeros(shape=(len(node_positions), len(relations)), dtype=np.long)
        for row in kb.itertuples():
            i = node_positions[row.e1]
            j = relation_positions[row.r]
            node_embeddings[i, j] = 1

    dot_product = node_embeddings.dot(node_embeddings.T)
    return dot_product, node_positions
