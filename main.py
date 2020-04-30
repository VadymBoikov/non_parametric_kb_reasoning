from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import time
import memory
from graph_utils import create_graph, is_relation_exist, get_nodes_by_relation

cutoff = 3
max_relations = 1000
top_k = 5
cores=12


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


def get_paths_reuse(G, nodes_reuse, relation, cases):
    memory_keys = []
    for node in nodes_reuse:
        related_nodes = get_nodes_by_relation(G[node], relation)
        for related in related_nodes:
            memory_keys.append((node, relation, related))

    path_counter = defaultdict(int)
    for key in memory_keys:
        node_paths = cases[key]
        for path in node_paths:
            path_counter[path] += 1

    sorted_paths = sorted([(v, k) for k, v in path_counter.items()], reverse=True)
    sorted_paths = [i[1] for i in sorted_paths]

    return sorted_paths


def get_nodes_at_path_end(G, start_node, path):
    end_nodes = []

    def recurse(path, i, node):
        if i == len(path):
            end_nodes.append(node)
            return

        relation = path[i]
        next_nodes = get_nodes_by_relation(G[node], relation)

        for next_node in next_nodes:
            recurse(path, i+1, next_node)

    recurse(path, 0, start_node)
    return end_nodes


def get_end_node(G, node, paths, count_answers=1):
    # paths should be sorted
    answers = []
    for path in paths:
        end_nodes = get_nodes_at_path_end(G, node, path)
        answers.extend(end_nodes)
        if len(answers) >= count_answers:
            return answers[:count_answers]

    return answers


def run_inference(q_node, q_relation, G, sim_mat, cases, count_answers=1, top_k=5):
    to_reuse = retreive_nodes(q_node, q_relation, sim_mat, node_ids, G, top_k=top_k)
    paths = get_paths_reuse(G, to_reuse, q_relation, cases)
    answer = get_end_node(G, q_node, paths, count_answers=count_answers)
    return answer


dataset_dir = 'data/WN18RR' # FB15k-237

print('load data')
kb = pd.read_csv(f'{dataset_dir}/train.txt', sep='\t', names=['e1', 'r', 'e2'])
G = create_graph(kb)

print('start cases')
cases = memory.create_memory_cases(kb, G, cutoff=cutoff, max_relations=max_relations, cores=cores)
print('start similarity')
sim_mat, node_ids = memory.create_similarity(G, sparse=False)
#
# print('dumping')
pickle.dump(cases, open(f'{dataset_dir}/memory_cases.pkl', 'wb'))
pickle.dump(node_ids, open(f'{dataset_dir}/node_ids.pkl', 'wb'))
np.save('data/sim_mat.npy', sim_mat)


# sim_mat = np.load('data/sim_mat.npy')
# cases = pickle.load(open('data/memory_cases.pkl', 'rb'))
# node_ids = pickle.load(open('data/node_ids.pkl', 'rb'))



kb_test = pd.read_csv('data/WN18RR/valid.txt', sep='\t', names=['e1', 'r', 'e2'])
print(len(kb_test))
correct = 0
er = 0

stats = []
for i in range(len(kb_test)):

    q_node1 = kb_test.iloc[i]['e1']
    q_relation = kb_test.iloc[i]['r']
    q_node2 = kb_test.iloc[i]['e2']
    if q_node1 not in G or q_node2 not in G:
        er += 1
        continue

    to_reuse = retreive_nodes(q_node1, q_relation, sim_mat, node_ids, G, top_k=top_k)
    paths = get_paths_reuse(G, to_reuse, q_relation, cases)
    answer = get_end_node(G, q_node1, paths, count_answers=1)

    is_correct = q_node2 in answer
    count_in_train = np.sum(kb.e1 == q_node1)
    stats.append([q_node1, q_relation, count_in_train, is_correct])
    if i % 100 == 0:
        print(i)

print(correct, er)



df = pd.DataFrame(stats, columns = ['e1', 'r', 'count', 'is_correct'])

z = df.groupby('count').apply(lambda x:x['is_correct'].value_counts())





