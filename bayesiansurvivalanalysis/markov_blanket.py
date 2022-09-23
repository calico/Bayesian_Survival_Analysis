import numpy as np
import itertools
import pymc3 as pm
from scipy.stats import chi2
from scipy.stats import norm
from scipy import stats
from copy import (copy,
                  deepcopy)
from bayes_network import Bayes_Net
from model import define_model
from utils import (remove_cycle_without_deletion,
                   remove_cycles,
                   starting_parameters)


def get_partial_matrix(S, X, Y):
    """ get correlation matrix gets rows and columns only correponding to the columns in  X and rows in Y
    :param S: covariance array
    :param X: contains list of indices corresponding to the two variables under consideration
    :param Y: the variables we want to condition on for this calculation
    :return S: partial matrix 
    """
    S = S[X, :]
    S = S[:, Y]
    return S


def partial_corr_coef(S, i, j, Y):
    """calculates partial correlation coefficient
    :param S: covariance array
    :param i: variable 1 to test conditional independence
    :param j: variable 2 to test for in conditional independence.
    :param Y: the variables we want to condition on for this calculation
    :return r: partial correlation coefficient of i and j given Y 
    """
    S = np.matrix(S)
    X = [i, j]
    inv_syy = np.linalg.inv(get_partial_matrix(S, Y, Y))
    i2 = 0
    j2 = 1
    S2 = get_partial_matrix(S, X, X) - (get_partial_matrix(S, X, Y) *
                                        inv_syy * get_partial_matrix(S, Y, X))
    c = S2[i2, j2]
    r = c / np.sqrt((S2[i2, i2] * S2[j2, j2]))
    return r


def cond_indep_fisher_z(data, var1, var2, cond=[], alpha=0.05):

    """
    COND_INDEP_FISHER_Z Test if var1 indep var2 given cond using Fisher's Z
    test
    CI = cond_indep_fisher_z(X, Y, S, C, N, alpha)
    C is the covariance (or correlation) matrix
    N is the sample size
    alpha is the significance level (default: 0.05)
    transfromed from matlab
    See p133 of T. Anderson, "An Intro. to Multivariate Statistical Analysis",
    1984
    :param data: pandas dataframe containing data
    :param var1: first variable in the independence condition.
    :param var2: second variable in the independence condition
    :param cond: List of variable names to condition on.
    :param alpha: significance level to test on.
    :return CI: The fisher z statistic for the test.
    :return r: partial correlation coefficient
    :return p_value: the p-value of the test
    """
    N, _ = np.shape(data)
    list_z = [var1, var2] + list(cond)
    list_new = []
    for a in list_z:
        list_new.append(int(a))
    data_array = np.array(data)
    array_new = np.transpose(np.matrix(data_array[:, list_new]))
    cov_array = np.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    z = 0.5 * np.log((1+r) / (1-r))
    z0 = 0
    W = np.sqrt(N - len(S1) - 3) * (z - z0)
    cutoff = norm.ppf(1 - 0.5 * alpha)
    if abs(W) < cutoff:
        CI = 1
    else:
        CI = 0
    p = norm.cdf(W)
    r = abs(r)
    return CI, r, p


def cond_indep_chi_square(data, var1, var2, cond=[], alpha=0.05):

    """
    COND_INDEP_FISHER_Z Test if var1 indep var2 given cond using chi square
    test
    CI = cond_indep_chi_square(X, Y, S, C, N, alpha)
    C is the covariance (or correlation) matrix
    N is the sample size
    alpha is the significance level (default: 0.05)
    :param data: pandas dataframe containing data
    :param var1: first variable in the independence condition.
    :param var2: second variable in the independence condition
    :param cond: List of variable names to condition on.
    :param alpha: significance level to test on.
    :return CI: The fisher z statistic for the test.
    :return r: partial correlation coefficient
    :return p_value: the p-value of the test
    """

    N, _ = np.shape(data)
    list_z = [var1, var2] + list(cond)
    list_new = []
    for a in list_z:
        list_new.append(int(a))
    data_array = np.array(data)
    array_new = np.transpose(np.matrix(data_array[:, list_new]))
    cov_array = np.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    t = r * np.sqrt((N - len(list_z))/(1 - (r * r)))
    pval = stats.t.sf(np.abs(t), N-1)*2
    if pval < alpha:
        CI = 1
    else:
        CI = 0
    r = abs(r)

    return CI, r, pval

    
def cond_indep_test(data, target, var, cond_set=[],
                    alpha=0.01, test='chisquare'):
    """conditional independence test.
    :param data: the data matrix to be used (as a numpy.ndarray).
    :param target: the first node (as an integer).
    :param var: the second node (as an integer).
    :param cond_set: the list of neibouring nodes of x and y (as a set()).
    :param alpha: p-value threshold for dependence.
    :param test: test type - implemented for chisquare and fisher Z test
    :return p_val: the p-value of conditional independence.
    :return dep: >0 (<0) if (not)significant p-value     
    """
    # continuous data under Chi-square test or Fisher Z test.
    # continuous data undergo chi-square test by default
    # Discrete data not implemented yet
    if test == 'chisquare':
        _, _, pval = cond_indep_chi_square(data, target, var, cond_set,
                                           alpha)
    else:
        _, _, pval = cond_indep_fisher_z(data, target, var, cond_set,
                                         alpha)

    if pval >= alpha:
        dep = - (1 - pval)
    else:
        dep = 1 - pval

    return pval, dep


def IAMB(data, target, alpha):
    """ IAMB performs incremental association Markov blanket for a given node
    :param data: pandas dataframe containing data
    :param target: target variable for which IAMB is performed.
    :alpha: significance level
    :return CMB: conditional Markov blanket for node
    :return ci_number: number of conditional independence test for dynamic p-value thresholding
    """
    _, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        # if not change, forward phase of IAMB is finished.
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y = None
        variables = [i for i in range(kVar) if i != target and i not in CMB]

        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, CMB)

            # chose maxsize of f(X:T|CMB)
            if pval <= alpha:
                if dep > temp_dep:
                    temp_dep = dep
                    y = x

        # if not condition independence the node,appended to CMB
        if y is not None:
            # print('appended is :'+str(y))
            CMB.append(y)
            circulate_Flag = True

    # backward circulate phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        # exclude variable which need test p-value
        condition_Variables = [i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(data, target, x, condition_Variables)
        # print("target is:", target, ",x is: ", x, " condition_Variables is: "
        #       , condition_Variables, " ,pval is: ", pval, " ,dep is: ", dep)
        if pval > alpha:
            # print("removed variables is: " + str(x))
            CMB.remove(x)

    return list(set(CMB)), ci_number


def resolve_markov_blanket(Mb, data, alpha=0.01):
    """
    Resolving the Markov blanket is the process
    by which a PDAG is constructed from the collection
    of Markov Blankets for each node. Since an
    undirected graph is returned, the edges still need to
    be oriented by calling some version of the
    "orient_edges" function.
    This algorithm is adapted from Margaritis.
    :param Mb: a dictionary, where
               key = rv and value = list of vars in rv's markov blanket
    :param data: data used to learn Markov blanket.
    :return edge_dict : a dictionaryof the resolved Markov blanet, where
                        key = rv and value = list of rv's children
    """
    n_rv = data.shape[1]
    edge_dict = dict([(rv, []) for rv in range(n_rv)])
    for X in range(n_rv):
        print('resolve markov blanket', X)
        for Y in Mb[X]:
            # X and Y are direct neighbors if X and Y are dependent
            # given S for all S in T, where T is the smaller of
            # B(X)-{Y} and B(Y)-{X}
            if len(Mb[X]) < len(Mb[Y]):
                T = copy(Mb[X])  # shallow copy is sufficient
                if Y in T:
                    T.remove(Y)
            else:
                T = copy(Mb[Y])  # shallow copy is sufficient
                if X in T:
                    T.remove(X)
            # X and Y must be dependent conditioned upon
            # EVERY POSSIBLE COMBINATION of T
            direct_neighbors = True
            for i in range(len(T)):
                for S in itertools.combinations(T, i):
                    pval, _ = cond_indep_test(data, X, Y, S)
                    if pval > alpha:
                        direct_neighbors = False
            if direct_neighbors:
                if Y not in edge_dict[X] and X not in edge_dict[Y]:
                    edge_dict[X].append(Y)
                if X not in edge_dict[Y]:
                    edge_dict[Y].append(X)
    return edge_dict


def orient_local_edges(need_to_resolve, data):
    """ orient local edges using WAIC from Bayesian regression models
    :param need_to_resolve: list of edges to orient
    :param data: dataframe with data
    :return direction_resolved: direction of edge based on local fits.
    """
    direction_resolved = []
    for curr_edge in need_to_resolve:
        A, B = curr_edge
        children_AB = {data.columns.values[A]: [data.columns.values[B]],
                       data.columns.values[B]: []}
        bn_AB = Bayes_Net(children_AB)
        model_AB = define_model(bn_AB, data, survival_layer=False, physiological_layer=True)
        with model_AB:
            start = starting_parameters(bn_AB)
            trace = pm.sample(1000, tune=1000, start=start, chains=2)
        waic_AB = pm.waic(trace, model_AB)
        children_BA = {data.columns.values[B]: [data.columns.values[A]],
                       data.columns.values[A]: []}
        bn_BA = Bayes_Net(children_BA)
        model_BA = define_model(bn_BA, data, survival_layer=False, physiological_layer=True)
        with model_BA:
            start = starting_parameters(bn_BA)
            trace = pm.sample(1000, tune=1000, start=start, chains=2)
        waic_BA = pm.waic(trace, model_BA)
        weight = (waic_BA.WAIC - waic_AB.WAIC) / (waic_BA.WAIC + waic_AB.WAIC)
        if weight > 0:
            direction_resolved.append((A, B))
        else:
            direction_resolved.append((B, A))
    return direction_resolved


def orient_edges(edge_dict, data):
    """ orient edges using global physiological network structure
    :param edge_dict: dictionary 
    :param data: dataframe with data
    :return d_edge_dict: resolved DAG
    """
    # Identifying list of edges that need to be resolved
    need_to_resolve = []
    d_edge_dict = {}
    for curr_parent in edge_dict:
        d_edge_dict[curr_parent] = []
        for curr_child in edge_dict[curr_parent]:
            # To ensure that this is only added once to need_to_resolve list_z
            if curr_child < curr_parent:
                continue
            if curr_parent in edge_dict[curr_child]:
                need_to_resolve.append((curr_parent, curr_child))
            else:
                d_edge_dict[curr_parent].append(curr_child)
    oriented_edges = orient_local_edges(need_to_resolve, data)
    for curr_edge in oriented_edges:
        d_edge_dict[curr_edge[0]].append(curr_edge[1])
    remove_cycle_without_deletion(d_edge_dict)
    remove_cycles(d_edge_dict)
    return d_edge_dict