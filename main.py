from collections import defaultdict
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import utils
import memory

cutoff= 3
max_relations = 1000
top_k = 10

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


def extract_nodes_reuse(G, search_relation, nodes_check=None, top_k=top_k):
    if nodes_check is None:
        nodes_check = G.nodes
    nodes_reuse = []
    for node_name in nodes_check:
        # check connected nodes for specific relation
        if is_relation_exist(G[node_name], search_relation):
            nodes_reuse.append(node_name)
        if len(nodes_reuse) == top_k:
            return nodes_reuse
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


def find_answer(G, node, relation, path_counter):
    
    sorted_paths = sorted([(v,k) for k,v in path_counter.items()], reverse=True)
    sorted_paths = [i[1] for i in sorted_paths]

    for path in sorted_paths:
        end_node = get_node_at_path_end(G, node, path)
        if end_node is not None:
            return end_node
    return None




kb = pd.read_csv('data/NELL-995/kb_env_rl.txt', sep='\t', names=['e1', 'e2', 'r'])
G = create_graph(kb)
cases = memory.create_memory_cases(kb, G)

pickle.dump(cases, open('data/memory_cases.pkl', 'wb'))

node_name = 'concept_politicaloffice_new'

len(G['concept_politicaloffice_new'])


c_key =  list(cases.keys())[1000]

cases[c_key]
sub = G.subgraph([node_name] + list(G[node_name].keys()))
nx.draw(sub)



sizes = []
for key in list(cases.keys()):
    sizes.append(len(cases[key]))

np.unique(sizes, return_counts=True)



