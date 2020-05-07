from collections import defaultdict

import numpy as np

import utils
from graph import is_relation_exist, get_nodes_by_relation


def retreive_nodes(q_node, q_relation, sim_mat, node_ids, G, top_k=5):
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
    :param paths: path to consider for query node. should be sorted by importance!!
    :param all_relations: because same relation may occure multiple times, we will not count them for evaluation
    :return:
    """

    rank = 0
    for path in paths:
        end_nodes = get_nodes_at_path_end(G, q_node, path)
        for end_node in end_nodes:
            if end_node == target_node:
                return rank
            if end_node not in all_relations:
                rank += 1
    return len(G.nodes)


@utils.timeit
def pipeline(valid, G, sim_mat, node_ids, cases, relations_kb, top_k=5, type='tail'):
    #TODO optimize to make in parallel without data move
    ranks = []
    for row in valid.itertuples():
        assert type in ['head', 'tail']
        q_node1, q_relation, q_node2 = row.e1, row.r, row.e2
        if type == 'head':
            q_node2, q_node1 = q_node1, q_node2

        to_reuse = retreive_nodes(q_node1, q_relation, sim_mat, node_ids, G, top_k=top_k)
        paths = get_paths_reuse(G, to_reuse, q_relation, cases)
        all_node_relations = relations_kb[(q_node1, q_relation)]
        rank = get_answer_rank(G, q_node1, q_node2, paths, all_node_relations)
        ranks.append(rank)
    return ranks