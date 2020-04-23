from collections import defaultdict
import multiprocessing as mp
import pickle
import time 

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import save_npz

import utils
import memory

cutoff = 4
max_relations = 1000
top_k = 5


def create_graph(kb):
    G = nx.DiGraph()
    for row in kb.itertuples():
        G.add_edge(row.e1, row.e2, relation = row.r)
    return G


def is_relation_exist(node, relation):
    for connected_node, attributes in node.items(): 
        if attributes['relation'] == relation:
            return True
    return False


def get_connected_by_relation(node, relation):
    related_nodes = []
    for connected_node, attributes in node.items(): 
        if attributes['relation'] == relation:
            related_nodes.append(connected_node)
    return related_nodes 


def retreive_nodes(q_node, q_relation, sim_mat, node_ids, G, top_k=10):
    position_names = {v:k for k,v in node_ids.items()}
    node_pos = node_ids[q_node]
    # check for sparsity
    if isinstance(sim_mat, np.ndarray):
        vec = sim_mat[node_pos]
    else:
        vec = sim_mat[node_pos].toarray()

    vec = vec[vec>0]
    order_ids = np.argsort(vec)[::-1]

    nodes_reuse = []
    for id in order_ids:
        check_node = position_names[id]
        if is_relation_exist(G[check_node], q_relation):
            nodes_reuse.append(check_node)
        if len(nodes_reuse) == top_k:
            break
    return nodes_reuse


def extract_paths(G, nodes_reuse, relation, cases):
    memory_keys = []
    for node in nodes_reuse:
        related_nodes = get_connected_by_relation(G[node], relation)
        for related in related_nodes:
            memory_keys.append((node, relation, related))
    
    path_counter = defaultdict(int)
    for key in memory_keys:
        node_paths = cases[key]
        for path in node_paths:
            path_counter[path] += 1
    
    return path_counter


def get_node_at_path_end(G, start_node, path):
    next_node = start_node
    for relation in path:
        connected = get_connected_by_relation(G[next_node], relation) 
        if len(connected):
            next_node = connected[0]  #TODO FIX. because I will extract only first occurence
        else:
            return None
    return next_node


def find_answer(G, node, path_counter):
    
    sorted_paths = sorted([(v,k) for k,v in path_counter.items()], reverse=True)
    sorted_paths = [i[1] for i in sorted_paths]

    for path in sorted_paths:
        end_node = get_node_at_path_end(G, node, path)
        if end_node is not None:
            return end_node
    return None


def enrich_inv(kb):
    kb_inv = []
    for row in kb.itertuples():
        kb_inv.append([row.e2, row.r, row.e1])
    kb_inv = pd.DataFrame(kb_inv, columns=kb.columns)
    out = pd.concat([kb, kb_inv])
    out = out.drop_duplicates().reset_index(drop=True)
    return out


print('load data')
kb = pd.read_csv('data/WN18RR/text/train.txt', sep='\t', names=['e1', 'r', 'e2'])
kb = enrich_inv(kb)
G = create_graph(kb)

# cases = pickle.load(open('data/memory_cases.pkl', 'rb'))
# sim_mat = pickle.load(open('data/memory_similarity.pkl', 'rb'))
print('start cases')
cases = memory.create_memory_cases(kb, G, cutoff=cutoff, max_relations=max_relations)
print('start similarity')
sim_mat, node_ids = memory.create_similarity(kb, sparse=False)
# pickle.dump(cases, open('data/memory_cases.pkl', 'wb'))
# pickle.dump(node_ids, open('data/node_ids.pkl', 'wb'))
# np.save('data/sim_mat.npy', similarity_matr)


kb_test = pd.read_csv('data/WN18RR/text/test.txt', sep='\t', names=['e1', 'r', 'e2'])
print(len(kb_test))
correct = 0
er = 0
for i in range(len(kb_test)):
    try:
        q_node1 = kb_test.iloc[i]['e1']
        q_relation = kb_test.iloc[i]['r']
        q_node2 = kb_test.iloc[i]['e2']

        to_reuse = retreive_nodes(q_node1, q_relation, sim_mat, node_ids, G, top_k=top_k)
        path_counter = extract_paths(G, to_reuse, q_relation, cases)
        answer = find_answer(G, q_node1, path_counter)
        if answer == q_node2:
            correct += 1
        if i % 100 == 0:
            print(i)
    except:
        er += 1
        continue

print(correct, er)



