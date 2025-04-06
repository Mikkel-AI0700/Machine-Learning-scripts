# WARNING: DO NOT COPY PASTE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import Union, List, Dict, Callable
import logging
import numpy
import pandas
import seaborn
import matplotlib.pyplot

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class InheritorClass:
    """
    Parent class that will be the class where the child classes will get the
    dataset, x variables, y_variables, and other keyword arguments

    Parameters:
        dataset (numpy.ndarray, pandas.DataFrame): Dataset to plot
        x_vars (Union[str, List[str]]): List of x variables to plot on x axis
        y_vars (Union[str, List[str]]): List of y variables to plot on y axis
    """
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None,
        x_vars: Union[str, List[str]] = None,
        y_vars: Union[str, List[str]] = None,
        **kwargs
    ):
        self.dataset = dataset
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.extra_params = kwargs

    def _calculate_row_amount (self, displot_type: bool = False, column_border: int = 3):
        """
        Protected method will calculate the amount of rows depending on plotting type

        Parameters:
            displot_type (bool): Set to True if plotting type is distributional
            column_border (int): The maximum amount of columns to create regardless of plot type

        Returns:
            row_amount (int): Amount of rows to create by plt.subplots
            element_length, column_border (int): Amount of columns to create by plt.subplots
        """
        if displot_type:
            element_length = len(self.x_vars)
        else:
            element_length = len(self.y_vars)

        row_amount = (element_length // column_border) + (element_length % column_border > 0)
        return row_amount, min(element_length, column_border)

    def _initialize_figure_axes (self, displot_type: bool = False):
        row_amount, column_amount = self._calculate_row_amount(displot_type=displot_type)
        figure, axes = matplotlib.pyplot.subplots(nrows=row_amount, ncols=column_amount, figsize=(25.5, 7.5))
        return figure, axes

    def _plot(self, axes = None, plot_method = None):
        axes = axes.flatten()

        if plot_method in [seaborn.histplot, seaborn.kdeplot, seaborn.ecdfplot]:
            for axes_iteration, column_iteration in enumerate(self.x_vars):
                plot_method(data=self.dataset, x=column_iteration, ax=axes[axes_iteration], **self.extra_params)
        else:
            for axes_iteration, column_iteration in enumerate(self.y_vars):
                plot_method(data=self.dataset, x=self.x_vars, y=column_iteration, ax=axes[axes_iteration], **self.extra_params)

class RelationalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None,
        x_vars: Union[str, List[str]] = None,
        y_vars: Union[str, List[str]] = None,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, **kwargs)
        self.relplot_methods = {
            "line": seaborn.lineplot,
            "scatter": seaborn.scatterplot
        }

    def plot_relational (self, relplot_type: str = None):
        figure, axes = self._initialize_figure_axes(displot_type=False)

        if relplot_type in self.relplot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.relplot_methods.get(relplot_type))
        else:
            raise ValueError("[!] Error: relplot argument doesn't exist in the relplot_methods dictionary")

class DistributionalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None,
        x_vars: Union[str, List[str]] = None,
        y_vars: Union[str, List[str]] = None,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, **kwargs)
        self.displot_methods = {
            "hist": seaborn.histplot,
            "kde": seaborn.kdeplot,
            "ecdf": seaborn.ecdfplot
        }

    def plot_distributional (self, displot_type: str = None):
        figure, axes = self._initialize_figure_axes(displot_type=True)

        if displot_type in self.displot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.displot_methods.get(displot_type))
        else:
            raise ValueError("[!] Error: displot argument doesn't exist in the displot_methods dictionary")


class CategoricalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None,
        x_vars: Union[str, List[str]] = None,
        y_vars: Union[str, List[str]] = None,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, **kwargs)
        self.catplot_methods = {
            "strip": seaborn.stripplot,
            "swarm": seaborn.swarmplot,
            "box": seaborn.boxplot,
            "boxen": seaborn.boxenplot,
            "bar": seaborn.barplot,
            "count": seaborn.countplot
        }

    def plot_categorical (self, catplot_type: str = None):
        figure, axes = self._initialize_figure_axes(displot_type=False)

        if catplot_type in self.catplot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.catplot_methods.get(catplot_type))
        else:
            raise ValueError("[!] catplot argument doesn't exist in the catplot methods dictionary")
