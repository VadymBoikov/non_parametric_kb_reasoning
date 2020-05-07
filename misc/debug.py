from main import *
from memory import make_relations_kb
from pipeline import retreive_nodes, get_paths_reuse, get_end_node, get_answer_rank

cutoff = 3
max_relations = 1000
top_k = 5
cores = 1


dataset_dir = '../data/WN18RR/text'  # FB15k-237

print('load data')
train = pd.read_csv(f'{dataset_dir}/train.txt', sep='\t', names=['e1', 'r', 'e2'])
valid = pd.read_csv(f'{dataset_dir}/valid.txt', sep='\t', names=['e1', 'r', 'e2'])


relations_kb = make_relations_kb(train, valid)
G = create_graph(train, add_inverse_name=True)
valid = valid.loc[valid.apply(lambda x: x.e1 in G and x.e2 in G, axis=1)]  # filter



print('start cases')
cases = memory.create_memory_cases(G, cutoff=cutoff, max_relations=max_relations, cores=cores)
print('start similarity')
sim_mat, node_ids = memory.create_similarity(G, sparse=False)


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

    all_relations = relations_kb[(q_node1, q_relation)]
    rank = get_answer_rank(G, q_node1, q_node2, paths, all_relations)
    x = get_end_node(G, q_node1, paths, count_answers=20)

    ranks.append(rank)

    if row.Index % 500 == 0:
        print(row.Index)

ranks = np.array(ranks)
ranks1, ranks3, ranks10 = sum(ranks <1), sum(ranks <3),  sum(ranks <10)
print(ranks1, ranks1/ len(valid), ranks3/ len(valid), ranks10/ len(valid))


#TODO CHECK WHAT IS WITH SIMILARITY, when rank >= 10 !!! (maybe answer my issue!)
np.argwhere(ranks==10)


