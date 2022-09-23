import pandas as pd
import argparse
import pymc3 as pm
from structure_learning import (univariate_analysis,
                                run_lasso,
                                run_mmhc,
                                run_physiological)
from model import define_model
from bayes_network import Bayes_Net
from utils import starting_parameters


def parse_args():
    """ 
    Parse arguments from Users.
    return: argument object with value of all user given arugments.
    """

    parser = argparse.ArgumentParser(description='Learn survival network and physiological network')
    parser.add_argument('--input-file',
                        help='file for input data in csv format.',
                        required=True),
    parser.add_argument('--output-prefix',
                        help='prefix for output files.',
                        required=True),
    parser.add_argument('--univariate',
                        action='store_true',
                        help='Perform univariate variable selection')
    parser.add_argument('--lasso',
                        action='store_true',
                        help='Perform LASSO based multivariate variable selection for alpha and beta')
    parser.add_argument('--mmhc',
                        action='store_true',
                        help='Perform MMHC search to identify variables associated with survival layer')
    parser.add_argument('--physiological',
                        action='store_true',
                        help='Perform IAMB to identify causal network in physiological layer')
    parser.add_argument('--time-column',
                        required=False,
                        default='time',
                        help='column with information of time to event (in years).\n Default="time" ')
    parser.add_argument('--status-column',
                        required=False,
                        default='status',
                        help='column with status informtion.\n Default="status" ')
    parser.add_argument('--adjustment-variables',
                        required=False,
                        nargs='+',
                        default=[],
                        help='the variables to adjust for during univariate variable test')
    args = parser.parse_args()
    return args


def main(input_file, output_prefix, univariate, lasso, mmhc, physiological, 
         adjustment_variables, time_col, status_col):
    """
    main function that performs structure learning for survival and/or physiological layers.
    :param input_file: name of input file with data in csv format.
    :param output_prefix: prefix for all output files.
    :param univariate: boolean variable that decides whether univariate analysis performed or not.
    :param lasso: boolean variable that decides whether lasso analysis performed or not.
    :param mmhc: boolean variable that decides whether MMHC based structure identification performed or not.
    :param physiological: boolean variable that decides whether structure identification of physiological layer performed or not.
    :param adjustment_variables: list of variables to adjust for in univariate analysis.
    :param time_col: column name in input data that contains time to event information.
    :param status_col: column name in input data that contains censoring information.
    """
    df = pd.read_csv(input_file)
    if not(univariate) and not(lasso) and not(mmhc) and not(physiological):
        mmhc = physiological = True
    if univariate:
        results = univariate_analysis(df, adjustment_variables, time_col, status_col)
        results.to_csv('{}_univariate_results.tsv'.format(output_prefix), sep='\t', index=False)
    if lasso:
        results = run_lasso(df, time_col, status_col)
        results.to_csv('{}_lasso_results.tsv'.format(output_prefix), sep='\t')
    if mmhc:
        c_dict, results = run_mmhc(df, time_col, status_col)
        results.to_csv('{}_mmhc_results.tsv'.format(output_prefix), sep='\t')
        with open('{}_survival_network.dat'.format(output_prefix), 'w') as op:
            op.write(str(c_dict))
    if physiological:
        children, results = run_physiological(df, time_col, status_col)
        results.to_csv('{}_physiological_results.tsv'.format(output_prefix), sep='\t')
        with open('{}_physiological_network.dat'.format(output_prefix), 'w') as op:
            op.write(str(children))
    if physiological and mmhc:
        # Merging both dictionaries
        children = {k: v + c_dict[k] for (k, v) in children.items()}
        children['logalpha'] = ['S']
        children['beta'] = ['S']
        children['S'] = []
        with open('{}_combined_network.dat'.format(output_prefix), 'w') as op:
            op.write(str(children))
        bn = Bayes_Net(children)
        model = define_model(bn, df, physiological_layer=True)
        begin = starting_parameters(bn)
        with model:
            trace = pm.sample(1000, tune=1000, start=begin, cores=4)
        results = pm.summary(trace)
        results.to_csv('{}_combined_results.tsv'.format(output_prefix), sep='\t')


if __name__ == '__main__':
    args = parse_args()
    main(args.input_file, args.output_prefix, args.univariate, args.lasso, args.mmhc, args.physiological,
         args.adjustment_variables, args.time_column, args.status_column)
