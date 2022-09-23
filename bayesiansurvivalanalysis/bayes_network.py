from utils import topological_sort


class Bayes_Net(object):
    """
    Overarching class for Bayesian Networks

    """

    def __init__(self, E=None):
        """Initialize the BayesNet class.
        :param E: a dict, where key = vertex, val = list of its children
        """
        if E is not None:
            # assert (value_dict is not None), 'Must set values if E is set.'
            self.set_structure(E)
        else:
            self.V = []
            self.E = {}
            self.F = {}

    def nodes(self):
        """Nodes in DAG
        :return v: each node in DAG
        """
        for v in self.V:
            yield v

    def children(self, rv):
        """Children of nodes in DAG
        :param rv: node representing random variable
        :return E[rv]: list of children of node in DAG
        """
        return self.E[rv]

    def set_structure(self, edge_dict):
        """
        Set the structure of a BayesNet object. This
        function is mostly used to instantiate a BN
        skeleton after structure learning algorithms.

        See "structure_learn" folder & algorithms

        :param edge_dict: a dictionary,
                          where key = rv,
                          value = list of rv's children
        """

        # topological sorting of graph
        self.V = topological_sort(edge_dict)
        self.E = edge_dict
        # set parents of nodes
        self.F = dict([(rv, {}) for rv in self.nodes()])
        for rv in self.nodes():
            self.F[rv] = {
                'parents': [p for p in self.nodes() if rv in self.children(p)],
                'cpt': [],
                'values': []
            }

    def parents(self, v):
        """
        Identify parents of each node in Bayes_Net object.
        :param v: node
        :return parents: parents of v
        """
        self.F[v]['parents'] = [n for n in self.nodes()
                                if v in self.children(n)]
        return self.F[v]['parents']

    def number_of_edges(self):
        """Number of edges in the network
        :return num_edges: number of edges in network
        """
        num_edges = 0
        for rv in self.nodes():
            num_edges += len(self.E[rv])
        return num_edges
