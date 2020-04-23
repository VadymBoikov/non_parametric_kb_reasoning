import numpy as np
import networkx as nx
import utils


def get_cases_relation(row, graph, cutoff, max_relations):
    # extract cases for one relation 
    path_generator = nx.all_simple_paths(graph, source=row.e1, target=row.e2, cutoff=cutoff)
    cases = set() # one node can have many equal relations
    for path in path_generator:
        relation_path = tuple([graph.get_edge_data(path[i], path[i+1])['relation'] for i in range(len(path)-1)])
        cases.add(relation_path)
        if len(cases) == max_relations:
            break
    return cases


#TODO iterate through graph, not knowledge base
def create_memory_cases(knowledge_base, graph, cutoff, max_relations):
    rows = [row for _, row in knowledge_base.iterrows()]
    relations = utils.run_function_on_list(get_cases_relation, rows, graph=graph,
                                            cutoff = cutoff, max_relations=max_relations)
    cases = {}
    for i, row in enumerate(knowledge_base.itertuples()):
        name = (row.e1, row.r, row.e2)
        cases[name] = relations[i]
    return cases


def create_similarity(kb, sparse=False):
    relations = sorted(np.unique(kb['r']))
    node_positions = {k:v for v,k in enumerate(sorted(np.unique(kb['e1'])))}
    relation_positions = {k:v for v,k in enumerate(relations)} 

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
