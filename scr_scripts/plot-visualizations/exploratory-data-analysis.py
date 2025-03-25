# WARNING: DO NOT COPY PASTE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import Union, List, Dict, Callable
import logging
import numpy
import pandas
import seaborn
from matplotlib.pyplot import subplots, Axes

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class InheritorClass:
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame],
        x_vars: Union[str, List[str]],
        y_vars: Union[str, List[str]],
        hue_color: str,
        **kwargs
    ):
        self.dataset = dataset
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.hue = hue_color
        self.extra_params = kwargs

    def _calculate_row_amount (self, column_border=3):
        column_amount = len(column_border)
        row_amount = (column_amount // column_border) + (column_amount % column_border > 0)
        return row_amount, column_amount

    def _initialize_figure_axes (self):
        row_amount, column_amount = self._calculate_row_amount()
        figure, axes = matplotlib.pyplot.subplots(nrows=row_amount, ncols=column_amount, figsize=(26.5, 15.5))
        return figure, axes

    def _plot(self, axes: Axes, plot_method: Callable[..., Axes]):
        if plot_method.__class__.__name__ in [method.__class__.__name__ for method in [seaborn.histplot, seaborn.kdeplot, seaborn.ecdfplot]]:
            for axes_iteration, column_iteration in enumerate(self.x_vars):
                plot_method(data=self.dataset, x=column_iteration, ax=axes[axes_iteration], **self.extra_params)
        else:
            for axes_iteration, column_iteration in enumerate(self.y_vars):
                plot_method(data=self.dataset, x=self.x_vars, y=column_iteration, ax=axes[axes_iteration], **self.extra_params)

class RelationalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame],
        x_vars: Union[str, List[str]],
        y_vars: Union[str, List[str]],
        hue_color: str,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, hue_color, **kwargs)
        self.relplot_methods = {
            "line": seaborn.lineplot,
            "scatter": seaborn.scatterplot
        }

    def plot_relational (self, relplot_type: str):
        figure, axes = self._initialize_figure_axes()

        if relplot_type in self.relplot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.relplot_methods.get(relplot_type))
        else:
            raise ValueError("[!] Error: relplot argument doesn't exist in the relplot_methods dictionary")

class DistributionalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame],
        x_vars: Union[str, List[str]],
        y_vars: Union[str, List[str]],
        hue_color: str,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, hue_color, **kwargs)
        self.displot_methods = {
            "hist": seaborn.histplot,
            "kde": seaborn.kdeplot,
            "ecdf": seaborn.ecdfplot
        }

    def plot_distributional (self, displot_type: str):
        figure, axes = self._initialize_figure_axes()

        if displot_type in self.displot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.displot_methods.get(displot_type))
        else:
            raise ValueError("[!] Error: displot argument doesn't exist in the displot_methods dictionary")


class CategoricalPlots (InheritorClass):
    def __init__ (
        self,
        dataset: Union[numpy.ndarray, pandas.DataFrame],
        x_vars: Union[str, List[str]],
        y_vars: Union[str, List[str]],
        hue_color: str,
        **kwargs
    ):
        super().__init__(dataset, x_vars, y_vars, hue_color, **kwargs)
        self.catplot_methods = {
            "strip": seaborn.stripplot,
            "swarm": seaborn.swarmplot,
            "box": seaborn.boxplot,
            "boxen": seaborn.boxenplot,
            "bar": seaborn.barplot,
            "count": seaborn.countplot
        }

    def plot_categorical (self, catplot_type: str):
        figure, axes = self._initialize_figure_axes()

        if catplot_type in self.catplot_methods.keys():
            logging.info("[*] Passing arguments to plot function...")
            self._plot(axes, self.catplot_methods.get(catplot_type))
        else:
            raise ValueError("[!] catplot argument doesn't exist in the catplot methods dictionary")