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


def get_end_node(G, q_node, paths, count_answers=1):
    # paths should be sorted
    answers = []
    for path in paths:
        end_nodes = get_nodes_at_path_end(G, q_node, path)

        for end_node in end_nodes:
            answers.append(end_node)
            if len(answers) >= count_answers:
                return answers[:count_answers]

    return answers


def get_answer_rank(G, q_node, target_node, paths, all_relations):
    """

    :param G:
    :param q_node:
    :param target_node:
    :param paths:
    :param all_relations: because same relation may occure multiple times, we will not count them for evaluation
    :return:
    """
    # paths should be sorted
    #TODO check how many paths exist per q_node. If many, than will be not effective

    rank = 0
    for path in paths:
        end_nodes = get_nodes_at_path_end(G, q_node, path)
        for end_node in end_nodes:
            if end_node == target_node:
                return rank
            if end_node not in all_relations:
                rank += 1
    return len(G.nodes)


def make_relations_kb(*datasets):
    # creates set of connections per (entity, relation), because there can multiple. So in evaluation we don't mix up
    vocab = defaultdict(set)
    for dataset in datasets:
        for row in dataset.itertuples():
            vocab[(row.e1, row.r)].add(row.e2)
    return vocab



if __name__ == '__main__':

    dataset_dir = 'data/WN18RR/text' # FB15k-237

    print('load data')
    train = pd.read_csv(f'{dataset_dir}/train.txt', sep='\t', names=['e1', 'r', 'e2'])
    valid = pd.read_csv(f'{dataset_dir}/valid.txt', sep='\t', names=['e1', 'r', 'e2'])

    relations_kb = make_relations_kb(train, valid)
    G = create_graph(train)
    valid = valid.loc[valid.apply(lambda x: x.e1 in G and x.e2 in G, axis=1)]  # filter
    print('start cases')
    cases = memory.create_memory_cases(G, cutoff=cutoff, max_relations=max_relations, cores=cores)
    print('start similarity')
    sim_mat, node_ids = memory.create_similarity(G, sparse=False)

    #
    # print('dumping')
    # pickle.dump(cases, open(f'{dataset_dir}/memory_cases.pkl', 'wb'))
    # pickle.dump(node_ids, open(f'{dataset_dir}/node_ids.pkl', 'wb'))
    # np.save('data/sim_mat.npy', sim_mat)


    # sim_mat = np.load('data/sim_mat.npy')
    # cases = pickle.load(open('data/memory_cases.pkl', 'rb'))
    # node_ids = pickle.load(open('data/node_ids.pkl', 'rb'))

    ranks = []
    for row in valid.itertuples():
        q_node1, q_relation, q_node2 = row.e1, row.r, row.e2
        to_reuse = retreive_nodes(q_node1, q_relation, sim_mat, node_ids, G, top_k=top_k)
        paths = get_paths_reuse(G, to_reuse, q_relation, cases)
        rank = get_answer_rank(G, q_node1, q_node2, paths, relations_kb)
        ranks.append(rank)

        if row.Index % 500 == 0:
            print(row.Index)