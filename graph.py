import networkx as nx


def create_graph(kb, add_inverse=True, add_inverse_name=True):
    G = nx.DiGraph()
    for row in kb.itertuples():
        if G.has_edge(row.e1, row.e2):
            G.get_edge_data(row.e1, row.e2)['relations'].add(row.r)
        else:
            G.add_edge(row.e1, row.e2, relations={row.r})

        if add_inverse:
            inv_r = row.r + '_inv' if add_inverse_name else row.r
            if G.has_edge(row.e2, row.e1):
                G.get_edge_data(row.e2, row.e1)['relations'].add(inv_r)
            else:
                G.add_edge(row.e2, row.e1, relations={inv_r})

    return G


def is_relation_exist(node, relation):
    for connected_node, attributes in node.items():
        if relation in attributes['relations']:
            return True
    return False


def get_nodes_by_relation(node, relation):
    # given nodes, return all adjacent nodes with given relation
    related_nodes = []
    for connected_node, attributes in node.items():
        if relation in attributes['relations']:
            related_nodes.append(connected_node)
    return related_nodes
