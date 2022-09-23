import pymc3 as pm
import pandas as pd
from bayes_network import Bayes_Net
from model import (define_model,
                   define_lasso_model)
from copy import deepcopy
from markov_blanket import (IAMB,
                            resolve_markov_blanket,
                            orient_edges)
from utils import (starting_parameters,
                   remove_cycles,
                   get_list_edges,
                   reverse_edge,
                   is_graph_cyclic)


def get_waic_model(bn, df, time_col, events_col, survival_layer=True, physiological_layer=False):
    """
    function that calculates WAIC for model
    :param bn: Bayes_Net object with DAG information
    :param df: pandas dataframe containing data
    :param time_col: column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :param survival_layer: does model contain survival layer?
    :param physiological_layer: does model contain physiological layer?
    :return WAIC: WAIC from model fit encoded by bn to the data.
    """
    model = define_model(bn, df, status_col=events_col, time_col=time_col,
                         survival_layer=survival_layer,
                         physiological_layer=physiological_layer)
    begin = starting_parameters(bn)
    with model:
        trace = pm.sample(1000, tune=1000, start=begin, cores=4)
    model_waic = pm.waic(trace, model)
    return model_waic.WAIC

def univariate_analysis(df, adjustment_variables, time_col, events_col):
    """
    function that performs univariate analysis for survival layer
    :param df: pandas dataframe containing data
    :param adjustment_variables: list of variables to adjust for in univariate analysis.
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :return results: results from all the univariate analysis on survival layer.
    """
    rel_variables = [curr_var for curr_var in df.columns.values
                     if curr_var not in adjustment_variables + [time_col, events_col]]
    c_dict = {}
    for curr_var in adjustment_variables:
        c_dict[curr_var] = ['logalpha', 'beta']
    c_dict['logalpha'] = ['S']
    c_dict['beta'] = ['S']
    c_dict['S'] =[]
    bn = Bayes_Net(c_dict)
    WAIC_base = get_waic_model(bn, df, time_col, events_col)
    results = pd.DataFrame(columns=['variable', 'diff_WAIC_alpha', 'diff_WAIC_beta'])
    for curr_var in rel_variables:
        edges_alpha, edges_beta = deepcopy(c_dict), deepcopy(c_dict)
        edges_alpha[curr_var] = ['logalpha']
        edges_beta[curr_var] = ['beta']
        bn = Bayes_Net(edges_alpha)
        WAIC_alpha = get_waic_model(bn, df[df[curr_var].notna()], time_col, events_col)
        bn = Bayes_Net(edges_beta)
        WAIC_beta = get_waic_model(bn, df[df[curr_var].notna()], time_col, events_col)
        results = results.append({'variable': curr_var,
                                  'diff_WAIC_alpha': WAIC_alpha - WAIC_base,
                                  'diff_WAIC_beta': WAIC_beta - WAIC_base},
                                 ignore_index=True)
    return results


def run_lasso(df, time_col, events_col):
    """
    function that performs LASSO analysis for survival layer
    :param df: pandas dataframe containing data
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :return summary: results from LASSO analysis on survival layer.
    """
    rel_variables = [curr_var for curr_var in df.columns.values
                     if curr_var not in [time_col, events_col]]
    network = {curr_var: ['logalpha', 'beta'] for curr_var in
               rel_variables}
    network['logalpha'] = ['S']
    network['beta'] = ['S']
    network['S'] = []
    bn = Bayes_Net(network)
    model = define_lasso_model(df, rel_variables, time_col, events_col)
    begin = starting_parameters(bn)
    with model:
        trace = pm.sample(1000, tune=1000, start=begin, cores=4)
    summary = pm.summary(trace)
    return summary


def initialize(df, time_col, events_col):
    """
    function that initializes calculations for MMHC search
    :param df: pandas dataframe containing data
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :return potential_relations: all the potential relations possible in graph.
    :return network: returns base network with no relations between variables and survival.
    """
    rel_variables = [curr_var for curr_var in df.columns.values
                     if curr_var not in [time_col, events_col]]
    network = {curr_var: [] for curr_var in rel_variables}
    network['logalpha'] = ['S']
    network['beta'] = ['S']
    network['S'] = []
    potential_relations = [(curr_var, 'logalpha') for curr_var in rel_variables]
    potential_relations += [(curr_var, 'beta') for curr_var in rel_variables]
    return potential_relations, network
    

def list_of_operations(bn, potential_relations=None):
    """
    function that calculates list of potential relations given current graph
    :param bn: Bayes_Net object with DAG information
    :return potential_relations: all the potential relations possible in graph.
                                 if potential_relations=None, all relations possible.
    :return operations: list of potential operations possible in a single interation to network.
    """
    operations = []
    c_dict = deepcopy(bn.E)
    for u in bn.nodes():
        for v in ['logalpha', 'beta']:
            # CHECK EDGE EXISTENCE 
            if (v not in c_dict[u]) and (u != v):
                # Edge Restrictions
                if potential_relations is None or \
                  (u, v) in potential_relations:
                    # SCORE FOR 'V' -> gaining a parent
                    operations.append(('Addition', u, v))
    for u in bn.nodes():
        for v in bn.nodes():
            if (u, v) == ('logalpha', 'S') or (u, v) == ('beta', 'S'):
                continue
            if (v in c_dict[u]):
                # SCORE FOR 'V' -> losing a parent
                if potential_relations is None or \
                  (u, v) in potential_relations:
                    operations.append(('Deletion', u, v))
    # No reversals due to survival layer alone
    return operations


def perform_single_operation(df, time_col, events_col, base_dict, curr_operation):
    """
    Calculates waic after applying single operation to graph
    :param df: pandas dataframe containing data
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :param base_dict: dictionary with base DAG information
    :param curr_operation: current operation in tuple form.
    :return potential_relations: all the potential relations possible in graph.
                                 if potential_relations=None, all relations possible.
    :return WAIC: WAIC from model fit encoded by graph encoded with single operation to base DAG.
    """
    children = deepcopy(base_dict)
    node1, node2 = curr_operation[1:]
    if curr_operation[0] == 'Addition':
        children[node1].append(node2)
    elif curr_operation[0] == 'Deletion':
        children[node1].remove(node2)
    bn = Bayes_Net(children)
    waic = get_waic_model(bn, df, time_col, events_col)
    return waic


def run_mmhc(df, time_col, events_col):
    """
    Max Min Hill cLimbing algorithm for optimizing structure of graph
    :param df: pandas dataframe containing data
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :return c_dict: optimized DAG
    :return summary: summary of parameters represented by optimized DAG.
    """
    potential_relations, c_dict = initialize(df, time_col, events_col)
    bn = Bayes_Net(c_dict)
    scores = {'Base': get_waic_model(bn, df, time_col, events_col)}
    improvement = True
    iter = 0
    while iter <= 1000 and improvement:
        operation_list = list_of_operations(bn, potential_relations=potential_relations)
        # Running all possible operations
        for curr_operation in operation_list:
            scores[curr_operation] = perform_single_operation(df, time_col, events_col, c_dict, curr_operation)    
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
        if sorted_scores[0][0] == 'Base':
            # No improvement in iteration
            improvement = False
        elif 'Addition' in sorted_scores[0][0]:
            # Add an edge to graph and run new iteration
            u = sorted_scores[0][0][1]
            v = sorted_scores[0][0][2]
            c_dict[u].append(v)
            scores = {'Base': sorted_scores[0][1]}
        elif 'Deletion' in sorted_scores[0][0]:
            # Delete an edge to graph and run new iteration
            u = sorted_scores[0][0][1]
            v = sorted_scores[0][0][2]
            c_dict[u].remove(v)
            scores = {'Base': sorted_scores[0][1]}
        bn = Bayes_Net(c_dict)
        iter += 1
    # Identify parameters for optimized DAG
    bn = Bayes_Net(c_dict)
    model = define_model(bn, df, status_col=events_col, time_col=time_col)
    begin = starting_parameters(bn)
    with model:
        trace = pm.sample(1000, tune=1000, start=begin, cores=4)
    summary = pm.summary(trace)
    return c_dict, summary


def hill_climbing_directionality(graph, data):
    """ Optimize direction of edges in graph using hill-climbing algorithm
    :param graph: graph to optimize
    :param data: data
    :return: improved graph
    """
    iter = 0
    improvement = True
    # Base graph
    var_graph = {}
    for curr_node in graph:
        var_graph[data.columns.values[curr_node]] = []
        for curr_child in graph[curr_node]:
            var_graph[data.columns.values[curr_node]].append(
                data.columns.values[curr_child]
            )
    remove_cycles(var_graph)
    bn = Bayes_Net(var_graph)
    scores = {'Base': get_waic_model(bn, data, time_col='Survival Time', events_col='status',
                                     survival_layer=False, physiological_layer=True)}
    tested_configurations = [var_graph]
    improvement = True
    iter = 0
    while iter <= 1000 and improvement:
        iter += 1
        # Identify edges that can be reversed
        list_edges = get_list_edges(var_graph)
        
        # Reverse each edge that does not result in cycle and calculate WAIC
        for idx_pair in range(len(list_edges)):
            edge = list_edges[idx_pair]
            test_graph = deepcopy(var_graph)
            reverse_edge(test_graph, edge[0], edge[1])
            if is_graph_cyclic(test_graph) \
               or test_graph in tested_configurations:
                continue
            bn = Bayes_Net(test_graph)
            curr_score = get_waic_model(bn, data, time_col='Survival Time', events_col='status',
                                        survival_layer=False, physiological_layer=True)
            scores[(edge[0], edge[1])] = curr_score
            tested_configurations.append(test_graph)
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
        print(sorted_scores)
        if sorted_scores[0][0] == 'Base':
            # No improvement possible
            improvement = False
        else:
            # reverse edge and start new iteration
            print(sorted_scores[0][0])
            u = sorted_scores[0][0][0]
            v = sorted_scores[0][0][1]
            print(sorted_scores, u, v)
            reverse_edge(var_graph, u, v)
            scores = {'Base': sorted_scores[0][1]}

    return var_graph


def run_physiological(df, time_col, events_col):
    """ Structure identification for physiological network
    :param df: pandas dataframe containing data
    :param time_col:  column name in input data that contains time to event information.
    :param events_col:  column name in input data that contains censoring information.
    :return c_dict: optimized physiological DAG
    :return summary: summary of parameters represented by optimized DAG.
    """
    rel_variables = [curr_var for curr_var in df.columns.values
                     if curr_var not in [time_col, events_col]]
    
    children = {}
    # Identify Markov blanket using IAMB
    for i in range(len(rel_variables)):
        MB, _ = IAMB(df[rel_variables], i, 0.01)
        children[i] = MB
    # Resolve Markov Blanket
    children = resolve_markov_blanket(children, df[rel_variables])
    # Orient edges locally and globally
    children = orient_edges(children, df[rel_variables])
    children = hill_climbing_directionality(children, df[rel_variables])
    # Identify parameters for optimized physiological DAG
    bn = Bayes_Net(children)
    model = define_model(bn, df, status_col=events_col, time_col=time_col,
                         survival_layer=False, physiological_layer=True)
    begin = starting_parameters(bn)
    with model:
        trace = pm.sample(1000, tune=1000, start=begin, cores=4)
    summary = pm.summary(trace)
    return children, summary
