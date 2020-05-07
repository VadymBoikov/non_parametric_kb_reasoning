import numpy as np
import pandas as pd
import copy
import graph
import networkx as nx


data_dir = '../data/WN18RR/original'
# data_dir = 'data/FB15k-237'

kb = pd.read_csv(f'{data_dir}/train.txt', sep='\t', names = ['e1', 'r', 'e2'])
valid = pd.read_csv(f'{data_dir}/valid.txt', sep='\t', names = ['e1', 'r', 'e2'])



#%% duplicated entity + relations in validation
dup_ids_e1 = valid[['e1', 'r']].duplicated()
dup_ids_e2 = valid[['e2', 'r']].duplicated()
print(f'valid shape {valid.shape}')
print(f"duplicated entity+relation in valid data e1+r:{dup_ids_e1.sum()}, e2+r:{dup_ids_e2.sum()}")

e1, r = valid.loc[dup_ids_e1, ['e1','r']].iloc[1]
print(valid.loc[(valid.e1 == e1) & (valid.r == r)]) # 8860123-> 15 9681351 -> 2


#%% check how many paths in valid already exist in train, but with different related entity
e1_e1 = pd.merge(kb, valid, how='inner', left_on=['e1', 'r'], right_on=['e1', 'r'])[['e1', 'r']].drop_duplicates()
e1_e2 = pd.merge(kb, valid, how='inner', left_on=['e1', 'r'], right_on=['e2', 'r'])[['e1_x', 'r']].drop_duplicates()
e2_e1 = pd.merge(kb, valid, how='inner', left_on=['e2', 'r'], right_on=['e1', 'r'])[['e1_y', 'r']].drop_duplicates()
e2_e2 = pd.merge(kb, valid, how='inner', left_on=['e2', 'r'], right_on=['e2', 'r'])[['e2', 'r']].drop_duplicates()

sh1, sh2, sh3,sh4 = e1_e1.shape[0], e1_e2.shape[0], e2_e1.shape[0], e2_e2.shape[0]
print(f"train e1 valid e1: {sh1}, train e1 valid e2: {sh2}, train e2 valid e1: {sh3}, train e2 valid e2: {sh4}")

e1, r = e1_e1.iloc[0][['e1', 'r']]
print(kb.loc[(kb.e1 == e1) & (kb.r == r)])
print(valid.loc[(valid.e1 == e1) & (valid.r == r)])  # 1-> 114, 0 ->34


#%%can we access valid e2 through others paths in train data

G = graph.create_graph(kb)
non_exist = []
for i, row in enumerate(valid.itertuples()):
    try:
        nx.all_simple_paths(G, row.e1, row.e2, cutoff=5).__next__()
    except StopIteration:
        non_exist.append(row)
    except AttributeError:
        non_exist.append(row)
    except nx.exception.NodeNotFound:
        continue
print(f"in validation with 5 steps relation doesnt exist in {len(non_exist)} cases")







train_counts = kb['e1'].value_counts()


valid_names = valid['e1'].unique()

train_counts.loc[train_counts.index.isin(valid_names)].value_counts()



kb2 = copy.deepcopy(kb)

joined = pd.merge(kb, kb2, how='outer', left_on=['e1', 'e2'], right_on=['e2', 'e1'])
inner = pd.merge(kb, kb2, how='inner', left_on=['e1', 'e2'], right_on=['e2', 'e1'])

left_nans = joined[joined['e1_x'].isna()]


not_equal_x = inner.loc[inner['r_x'] != inner['r_y'], 'r_x']
not_equal_y = inner.loc[inner['r_x'] != inner['r_y'], 'r_y']


equal_relations = inner.loc[inner['r_x'] == inner['r_y'], 'r_x']

np.unique(kb.r, return_counts=True)
np.unique(equal_relations, return_counts=True)
np.unique(not_equal_x, return_counts=True)


kb.loc[kb[['e1', 'e2']].duplicated()].sort_values(by=['e1', 'e2'])['r'].unique()

kb.loc[(kb.e1 == 'abandon.v.01') & (kb.e2 == 'abandonment.n.03')]


set(kb['e2']) - set(kb['e1'])

train.shape

valid.shape



total = pd.concat([train, valid])
total.drop_duplicates().shape

total.shape

len(set(valid['e1']) - set(train['e1']))