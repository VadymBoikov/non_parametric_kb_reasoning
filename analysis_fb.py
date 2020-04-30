import numpy as np
import pandas as pd
import copy
import graph_utils as gutils
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


kb = pd.read_csv('data/FB15k-237/train.txt', sep='\t', names = ['e1', 'r', 'e2'])
valid = pd.read_csv('data/WN18RR/valid.txt', sep='\t', names = ['e1', 'r', 'e2'])


G = gutils.create_graph(kb, add_inverse=True)
G2 = gutils.create_graph(kb, add_inverse=False)



print(kb.shape)


def plot_frequency(col, percentile=0.99):
    # how often values are being reused
    counts = col.value_counts().value_counts().sort_index()

    cum_percent = np.cumsum(counts) / sum(counts)
    counts = counts.loc[cum_percent <= percentile]

    sns.lineplot(x = counts.index, y = counts)


plot_frequency(kb['e1'])
plot_frequency(kb['e2'])
plot_frequency(col=kb['r'], percentile=0.9)


def plot_multi_relations(G, log_scale=False):
    count_relations = defaultdict(int)
    for (u,v) in G.edges:
        count =  len(G.get_edge_data(u,v)['relations'])
        count_relations[count] +=1

    count_relations = pd.Series(count_relations).sort_index()

    if log_scale:
        count_relations = np.log2(count_relations)
    sns.barplot(x = count_relations.index, y = count_relations)


plot_multi_relations(G, log_scale=True)



