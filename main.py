import argparse
import multiprocessing as mp
import os
import numpy as np
import pandas as pd

import memory
from graph import create_graph
from pipeline import pipeline

cutoff = 3
max_relations = 1000
top_k = 5
cores = mp.cpu_count()


def print_scores(ranks_tail, ranks_head):
    size = len(ranks_tail)
    tail, head = np.array(ranks_tail), np.array(ranks_head)

    for i in [1, 3, 10]:
        hit_tail, hit_head = sum(tail < i) / size, sum(head < i) / size
        hit = np.mean([hit_tail, hit_head])

        mrr_tail, mrr_head = np.mean(1 / (tail + 1)), np.mean(1 / (head + 1))
        mrr = np.mean([mrr_tail, mrr_head])

        hit, mrr = np.round(hit, 2), np.round(mrr, 2)
        print(f"hit @{i}: {hit}, MRR: {mrr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', help='name of the dataset', default='WN18RR')
    args = parser.parse_args()
    dataset = args.dataset

    train_dir = os.path.join("data", dataset, "train.txt")
    valid_dir = os.path.join("data", dataset, "valid.txt")

    print(f'load dataset {dataset}')
    train = pd.read_csv(train_dir, sep='\t', names=['e1', 'r', 'e2'])
    valid = pd.read_csv(valid_dir, sep='\t', names=['e1', 'r', 'e2'])

    relations_kb = memory.make_relations_kb(train, valid)
    G = create_graph(train, add_inverse_name=False)
    valid = valid.loc[valid.apply(lambda x: x.e1 in G and x.e2 in G, axis=1)]  # filter nodes, that not present in train

    print('start cases')
    cases = memory.create_memory_cases(G, cutoff=cutoff, max_relations=max_relations, cores=cores)
    print('start similarity')
    sim_mat, node_ids = memory.create_similarity(G, sparse=True)

    print('start inference')
    ranks_tail = pipeline(valid, G, sim_mat, node_ids, cases, relations_kb=relations_kb, top_k=top_k, type='tail')
    ranks_head = pipeline(valid, G, sim_mat, node_ids, cases, relations_kb=relations_kb, top_k=top_k, type='head')

    print_scores(ranks_tail, ranks_head)
