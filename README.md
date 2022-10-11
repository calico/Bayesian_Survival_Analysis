This repository performs the causal structure discovery of variables associated with survival using Gompertz-based parametric survival models. For more details, please read the associated manuscript [1]. The independent variables need to be provided as a csv file along with the survival data. In addition to causal structure discovery in the survival layer, the code can also be used to perform join inference of physiological network (IAMB) and survival layer (MMHC) to identify both direct and indirect effects on aging.

# Getting Started

The recommended build environment for the code is to have Anaconda installed and then to create a conda environment for python 3 as shown below:

> conda create -n csd python=3.6.8

> conda activate csd

> pip install -r requirements.txt

# Running the code

Once the environment is setup and the input file (csv) is ready, you can use the script "perform_learning.py" to perform causal structure discovery

> python perform_learning.py --help

usage: perform_learning.py [-h] --input-file INPUT_FILE --output-prefix
                           OUTPUT_PREFIX [--univariate] [--lasso] [--mmhc]
                           [--physiological] [--time-column TIME_COLUMN]
                           [--status-column STATUS_COLUMN]
                           [--adjustment-variables ADJUSTMENT_VARIABLES [ADJUSTMENT_VARIABLES ...]]

Learn survival network and physiological network

optional arguments:
  -h, --help            show this help message and exit

  --input-file INPUT_FILE
                        file for input data in csv format.
  
  --output-prefix OUTPUT_PREFIX
                        prefix for output files.

  --univariate          Perform univariate variable selection
  
  --lasso               Perform LASSO based multivariate variable selection
                        for alpha and beta
  
  --mmhc                Perform MMHC search to identify variables associated
                        with survival layer
  
  --physiological       Perform IAMB to identify causal network in
                        physiological layer
  
  --time-column TIME_COLUMN
                        column with information of time to event (in years).
                        Default="time"
  
  --status-column STATUS_COLUMN
                        column with status informtion. Default="status"
  
  --adjustment-variables ADJUSTMENT_VARIABLES [ADJUSTMENT_VARIABLES ...]
                        the variables to adjust for during univariate variable
                        test


To perform the causal structure discovery methods highlighted in the article [1], please run:

> python perform_learning.py --input-file [input_file.csv] --output-prefix [output_prefix] --mmhc --physiological

# Output files

## Univariate Analysis
If you perform univariate analysis, you will see the file:
[outputprefix]_univariate_results.tsv - this file is tab separated with 3 columns in the header: variable, diff_WAIC_alpha, and diff_WAIC_beta. Each row in the file represents the results for a single variable.  diff_WAIC_alpha (diff_WAIC_beta) represents the difference in WAIC for fit of variable to alpha (beta).

## LASSO Analysis
IF you perform LASSO analysis, you will see the file:

[outputprefix]_lasso_results.tsv - this file is tab separated and contains the summary of the results from the pymc3 fit of the LASSO model. Each row in the file represents the results for a single variable on alpha or beta. The row represented by "a_[variablename]" represents the parameter for effect of the variable on alpha while "b_[variablename]" represents the parameter for effect of the variable on beta. The posterior densities represented by hpd_2.5 and hpd_97.5 should not intersect with null effect size (0) to be considered significant.

## MMHC Analysis
IF you perform MMHC analysis, you will see the files:

[outputprefix]_survival_network.dat - this file contains the structure of the survival layer in dictionary format. The keys represent each individual node in the network while the values represent the list of children nodes of each variable. In the survival network, the variables can only be connected to alpha and/or beta while alpha and beta are connected to the survival node S.

[outputprefix]_mmhc_results.tsv - this file is tab separated and contains the summary of the results from the pymc3 fit of the MMHC model. Each row in the file represents the results for a single variable on alpha or beta after variable selection. The row represented by "a_[variablename]" represents the parameter for effect of the variable on alpha while "b_[variablename]" represents the parameter for effect of the variable on beta. 


## Physiological Analysis
If you perform physiological analysis, you will see the files:

[outputprefix]_physiological_network.dat - this file contains the structure of the physiological layer in dictionary format. The keys represent each individual node in the network while the values represent the list of children nodes of each variable. 

[outputprefix]_physiological_results.tsv - this file is tab separated and contains the summary of the results from the pymc3 fit of the MMHC model. Each row in the file represents the effect size estimation for parents on children after causal structure discovery. The row represented by "[variable1]_[variable2]" encodes the effect size of variable 2 on variable 1 (i.e., variable 2 is the parent of variable 1 in the network). Causal structure discovery is done using IAMB method as explained in [1].

## Combined Analysis
If you perform both MMHC and physiological analyses together, you will see the files:

[outputprefix]_combined_network.dat - this file contains the structure of the physiological and suvival layers together in dictionary format. The keys represent each individual node in the network while the values represent the list of children nodes of each variable. 

[outputprefix]_combined_results.tsv - this file is tab separated and contains the summary of the results from the pymc3 fit of the MMHC model. Each row in the file represents the effect size estimation of parents on children within the combined network.

DOI for citing software: 10.5281/zenodo.7186584

[1] Joint inference of physiological network and survival analysis identifies factors associated with aging rate. Sethi and Melamud, Cell Reports Methods, 2022, Accepted.

