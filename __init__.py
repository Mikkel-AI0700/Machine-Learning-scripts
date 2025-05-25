from setuptools import setup, find_packages

from scr_scripts.data_preprocessing_scripts.encoding_scaling_scripts import *
from scr_scripts.data_preprocessing_scripts.null_outlier_scripts import *
from scr_scripts.data_preprocessing_scripts.skewed_kurtosis_scripts import N

from scr_scripts.dataset_loader import loader
from scr_scripts.dependency_importer import dependency_importer
from scr_scripts.hyperparameter_tuning_scripts import random_search, grid_search
from scr_scripts.plot_visualizations import exploratory_data_analysis, show_eval_plot

setup(
    name = "autosklearn",
    version = 0.1,
    packages = find_packages(),
    install_requires = [
        "numpy",
        "pandas",
        "seaborn"
        "matplotlib",
        "matplotlib.pyplot"
    ]
)
