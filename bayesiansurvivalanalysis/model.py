import pymc3 as pm
import numpy as np


def survival(event, logalpha, beta, time):
    """
    function calculates the log likelihood of Gompertz models
    :param event: vector that contains censoring information
    :param logalpha: pymc3 vector that contains log(alpha) Gompertz parameter
    :param beta: pymc3 vector that contains beta Gompertz parameter
    :param time: vector that contains time to event information
    :return: vector containing loglikelihood of data given parameters
    """
    log_h = logalpha + beta*time
    H = (np.exp(logalpha)/beta) * (np.exp(beta*time) - 1)
    return ((event * log_h) - H).sum()


def define_model(bn, data, status_col='status', time_col='Survival Time',
                 save_all=True, survival_layer=True, physiological_layer=False):
    """
    function that creates pymc3 Gompertz survival model
    :param bn: Bayes_Net object with DAG information
    :param data: pandas dataframe containing data
    :param status_col:  column name in input data that contains censoring information.
    :param time_col:  column name in input data that contains time to event information.
    :param save_all: boolean flag that decides whether everything is saved in pymc3 or only parameters.
                     Set it to False to save memory from saving logalpha and beta.
    :param survival_layer: does model contain survival layer?
    :param physiological_layer: does model contain physiological layer?
    :return model: pymc3 model containing all parameters encoded by bn.
    """             
    BoundedHalfNormal = pm.Bound(pm.HalfNormal, lower=0, upper=2)
    with pm.Model() as model:
        if survival_layer:
            # logalpha covariates
            a_sigma_j, a_coeff, b_sigma_j, b_coeff = [], [], [], []
            BoundedNormal = pm.Bound(pm.Normal, lower=-30, upper=0)
            BoundedNormal2 = pm.Bound(pm.Normal, lower=0, upper=0.5)
            BoundedNormal3 = pm.Bound(pm.Normal, lower=-3, upper=3)
            mu_a = BoundedNormal('a0', mu=0, sd=30)
            a_sigma_s = BoundedHalfNormal('a_sigma_s', sd=10)
            for i, curr_node in enumerate(bn.F['logalpha']['parents']):
                a_sigma_j.append(BoundedHalfNormal('a_sigma_j_{}'
                                                   .format(curr_node),
                                                   sd=10))
                a_coeff.append(BoundedNormal3('a_{}'.format(curr_node),
                                              mu=0,
                                              sd=a_sigma_s*a_sigma_j[-1]
                                             )
                              )
                mu_a += a_coeff[i] * data[curr_node]
            if save_all:
                logalpha = pm.Deterministic('logalpha', mu_a)
            else:
                logalpha = mu_a
            mu_b = BoundedNormal2('b0', sd=10)
            b_sigma_s = BoundedHalfNormal('b_sigma_s', sd=10)
            for i, curr_node in enumerate(bn.F['beta']['parents']):
                # beta covariates
                b_sigma_j.append(BoundedHalfNormal('b_sigma_j_{}'
                                                   .format(curr_node),
                                                   sd=10
                                                  )
                                )
                b_coeff.append(BoundedNormal3('b_{}'.format(curr_node),
                                              mu=0,
                                              sd=b_sigma_s*b_sigma_j[-1]
                                              )
                              )
                mu_b += b_coeff[i] * data[curr_node]
            if save_all:
                beta = pm.Deterministic('beta', mu_b)
            else:
                beta = mu_b
            # overall survival
            S = pm.DensityDist('S', survival,
                               observed={'event': data[status_col],
                                         'logalpha': logalpha,
                                         'beta': beta,
                                         'time': data[time_col]
                                        }
                              )
        if physiological_layer:
            # linear regression of different physiological variables
            param, mu, y, sigma, intercept = {}, {}, {}, {}, {}
            for curr_feature in bn.nodes():
                if curr_feature in ['logalpha', 'beta', 'S']:
                    continue
                param[curr_feature] = {}
                intercept[curr_feature] = pm.Normal('{}_intercept'
                                                    .format(curr_feature),
                                                    mu=0,
                                                    sd=1)
                m = intercept[curr_feature]
                for curr_node in bn.F[curr_feature]['parents']:
                    # covariates of each physiological variable
                    param[curr_feature][curr_node] = \
                        pm.Normal('{}_{}'.format(curr_feature, curr_node),
                                  mu=0,
                                  sd=10
                                  )
                    m += param[curr_feature][curr_node] * \
                        data[curr_node]
                mu[curr_feature] = m
                sigma[curr_feature] = BoundedHalfNormal('{}_sigma'
                                                        .format(curr_feature),
                                                        sd=10)
                y[curr_feature] = pm.Normal(curr_feature,
                                            mu=mu[curr_feature],
                                            sd=sigma[curr_feature],
                                            observed=data[curr_feature])

    return model


def define_lasso_model(data, variables, time_col, status_col):
    """
    function that creates pymc3 Gompertz survival model for lasso model
    :param data: pandas dataframe containing data
    :param variables: list of variables to consider in LASSO model.
    :param time_col:  column name in input data that contains time to event information.
    :param status_col:  column name in input data that contains censoring information.
    :return model: pymc3 model containing all parameters encoded by the LASSO model.
    """
    with pm.Model() as model:
        a_coeff, b_coeff = [], []
        BoundedNormal1 = pm.Bound(pm.Normal, lower=-30, upper=0)
        BoundedNormal2 = pm.Bound(pm.Normal, lower=0, upper=0.5)
        BoundedLaplace1 = pm.Bound(pm.Laplace, lower=-3, upper=3)
        BoundedLaplace2 = pm.Bound(pm.Laplace, lower=-0.1, upper=0.1)
        a0 = BoundedNormal1('a0', mu=0, sd=30)
        b0 = BoundedNormal2('b0', sd=1)
        mu_a = a0
        mu_b = b0
        lam = pm.HalfCauchy('lambda', 1)
        for (i, curr_var) in enumerate(variables):
            a_coeff.append(
                BoundedLaplace1('a_{}'.format(curr_var), mu=0, b=1/lam))
            b_coeff.append(
                BoundedLaplace2('b_{}'.format(curr_var), mu=0, b=1/lam))
            mu_a += a_coeff[i] * data[curr_var]
            mu_b += b_coeff[i] * data[curr_var]
        logalpha = mu_a
        beta = mu_b
        S = pm.DensityDist('S', survival,
                           observed={'event': data[status_col],
                                     'logalpha': logalpha,
                                     'beta': beta,
                                     'time': data[time_col]
                                     }
                           )
    return model
