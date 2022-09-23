import networkx as nx


def topological_sort_util(edge_dict, v, visited, stack):
    """
    topological sorting util sorts the nodes in graph topologically for each vertex
    :param edge_dict: graph
    :param v: node
    :visited: boolean vector representing whether node was previously visited or not.
    :stack: stack that stores position of each vertex for topological storing
    """
    # Mark the current node as visited.
    i = list(edge_dict.keys()).index(v)
    if visited[i]:
        return
    visited[i] = True
    # Recur for all the vertices adjacent to this vertex
    for n in edge_dict[v]:
        i = list(edge_dict.keys()).index(n)
        if visited[i] is False:
            topological_sort_util(edge_dict, n, visited, stack)
    # Push current vertex to stack which stores result
    stack.insert(0, v)


def topological_sort(edge_dict):
    """
    topological sorting sorts a given graph
    :param edge_dict: graph
    :return stack: order of vertices after topological sorting.
    """
    # Based upon code from
    # https://www.geeksforgeeks.org/topological-sorting/
    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    # Mark all the vertices as not visited
    visited = [False]*len(edge_dict)
    stack = []
    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for i in edge_dict.keys():
        topological_sort_util(edge_dict, i, visited, stack)
    return stack


def would_cause_cycle(e, u, v, reverse=False):
    """
    Test if adding the edge u -> v to the BayesNet
    object would create a DIRECTED (i.e. illegal) cycle.
    :param e: edge dictionary 
    :param u: node u
    :param v: node v
    :param reverse: does reversal of v -> u to u -> v cause a cycle?
    :return : boolean representing whether it causes a cycle
    """
    G = nx.DiGraph(e)
    if reverse:
        G.remove_edge(v, u)
    G.add_edge(u, v)
    try:
        nx.find_cycle(G, source=u)
        return True
    except:
        return False


def reverse_edge(g, node1, node2):
    """ Reverse the edge between node1 and node2
    :param g: graph represented as a dictionary
    :param node1: initial cause of the edge
    :param node2: initial effect of the edge
    """
    g[node1].remove(node2)
    g[node2].append(node1)


def remove_cycle_without_deletion(g):
    """
    resolve cycles without deletion of edges
    :param g: graph represented as a dictionary mapping vertices to
              iterables of neighbouring vertices.
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path:
                reverse_edge(g, vertex, neighbour)
            else:
                visit(neighbour)
        path.remove(vertex)

    for v in g:
        visit(v)


def remove_cycles(g):
    """
    Remove cycles by removing edges and return DAG
    graph g is updated
    :param g: graph represented as a dictionary mapping vertices to
              iterables of neighbouring vertices.
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path:
                g[vertex].remove(neighbour)
            else:
                visit(neighbour)
        path.remove(vertex)

    for v in g:
        visit(v)


def starting_parameters(bn):
    """starting parameters for pymc3
    :param bn: Bayes_Net object contaiing graph
    :return params: dictionary of starting parameters for MCMC walk in pymc3
    """
    params = {}
    for curr_var in bn.nodes():
        if curr_var not in ['logalpha', 'beta', 'S']:
            params['mu_{}'.format(curr_var)] = 0
            params['{}_sigma'.format(curr_var)] = 1
        elif curr_var == 'logalpha':
            params['a_sigma_s'] = 1
            for curr_parent in bn.F['logalpha']['parents']:
                params['a_sigma_j_{}'.format(curr_parent)] = 0.1
                params['a_{}'.format(curr_parent)] = 0
        elif curr_var == 'beta':
            params['b0'] = 0.1
            params['b_sigma_s'] = 0.1
            for curr_parent in bn.F['beta']['parents']:
                params['b_sigma_j_{}'.format(curr_parent)] = 0.1
                params['b_{}'.format(curr_parent)] = 0
    return params


def is_graph_cyclic(g):
    """
    Return True if the directed graph g has a cycle.
    :param g: g must be represented as a dictionary mapping vertices to
              iterables of neighbouring vertices.
    :return: True if the directed graph is cyclic
    :rtype: bool
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)


def get_list_edges(g):
    """ Get list of edges according to order defined by parameters
    :param g: dictionary graph with nodes as keys and
              value being list of children from node
    :return: List of edges
    """
    list_edges = []
    for i in g:
        for j in list(g[i]):
            list_edges.append([i, j])

    return list_edges