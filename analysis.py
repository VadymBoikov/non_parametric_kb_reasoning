import numpy as np
import pandas as pd
import copy


kb = pd.read_csv('data/WN18RR/train.txt', sep='\t', names = ['e1', 'r', 'e2'])
valid = pd.read_csv('data/WN18RR/valid.txt', sep='\t', names = ['e1', 'r', 'e2'])



kb = enrich_inv(kb)


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