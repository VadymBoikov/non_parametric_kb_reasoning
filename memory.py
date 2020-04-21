import networkx as nx
import utils

def get_cases_relation(row, graph, cutoff=cutoff, max_relations=max_relations):
    # extract cases for one relation 
    path_generator =  nx.all_simple_paths(graph, source=row.e1, target=row.e2, cutoff=cutoff)
    cases = set() # one node can have many equal relations
    for path in path_generator:
        relation_path = tuple([graph.get_edge_data(path[i], path[i+1])['relation'] for i in range(len(path)-1)])
        cases.add(relation_path)
        if len(cases) == max_relations:
            break
    return cases


#TODO iterate through graph, not knowledge base
def create_memory_cases(knowledge_base, graph):
    rows = [row for _, row in knowledge_base.iterrows()]
    relations = utils.run_function_on_list(get_cases_relation, rows, graph=graph)
    cases = {}
    for i, row in enumerate(knowledge_base.itertuples()):
        name = (row.e1, row.r, row.e2)
        cases[name] = relations[i]
    return cases
