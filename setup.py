from distutils.core import setup

setup(
    name='BayesianSurvivalAnalysis',
    version='0.1dev',
    packages=['bayesiansurvivalanalysis',],
    long_description=open('README.txt').read(),
    install_requires=[
        "pymc3 >= 3.5",
        "pandas >= 0.23.4",
        "networkx >= 2.4",
        "arviz >= 0.5.1",
    ],
)
