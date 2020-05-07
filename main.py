import numpy as np
import pandas as pd
import memory
from graph import create_graph
from pipeline import pipeline

cutoff = 3
max_relations = 1000
top_k = 5
cores = 12


def print_scores(ranks_tail, ranks_head):
    size = len(ranks_tail)
    tail, head = np.array(ranks_tail), np.array(ranks_head)

    for i in [1, 3, 10]:
        hit_tail, hit_head = sum(tail < i) / size, sum(head < i) / size
        hit = np.mean([hit_tail, hit_head])
        hit_tail, hit_head, hit = round(hit_tail, 2), round(hit_head,2), np.round(hit, 2)
        print(f"hit @{i}: tail:{hit_tail}, head:{hit_head}, total:{hit}")


if __name__ == '__main__':
    dataset_dir = "data/FB15k-237"  # 'data/WN18RR/original' #

    print('load data')
    train = pd.read_csv(f'{dataset_dir}/train.txt', sep='\t', names=['e1', 'r', 'e2'])
    valid = pd.read_csv(f'{dataset_dir}/valid.txt', sep='\t', names=['e1', 'r', 'e2'])

    relations_kb = memory.make_relations_kb(train, valid)
    G = create_graph(train, add_inverse_name=True)
    valid = valid.loc[valid.apply(lambda x: x.e1 in G and x.e2 in G, axis=1)]  # filter nodes, that not present in train

    print('start cases')
    cases = memory.create_memory_cases(G, cutoff=cutoff, max_relations=max_relations, cores=cores)
    print('start similarity')
    sim_mat, node_ids = memory.create_similarity(G, sparse=True)

    ranks_tail = pipeline(valid, G, sim_mat, node_ids, cases, relations_kb=relations_kb, type='tail')
    ranks_head = pipeline(valid, G, sim_mat, node_ids, cases, relations_kb=relations_kb, type='head')

    print_scores(ranks_tail, ranks_head)
