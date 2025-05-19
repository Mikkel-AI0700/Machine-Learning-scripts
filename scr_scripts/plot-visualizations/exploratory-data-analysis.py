
from typing import Callable

import numpy
import pandas
import seaborn
import matplotlib

class PlotDataset:
    def __init__ (self):
        self.relplot_functions = {
            "scatter": seaborn.scatterplot,
            "line": seaborn.lineplot
        }
        self.displot_functions = {
            "hist": seaborn.histplot,
            "kde": seaborn.kdeplot,
            "ecdf": seaborn.ecdfplot
        }
        self.catplot_functions = {
            "strip": seaborn.stripplot,
            "box": seaborn.boxplot,
            "bar": seaborn.barplot,
            "count": seaborn.countplot
        }

    def _calculate_row_amount (
        self, 
        x_vars: list[str, ...],
        y_vars: list[str, ...],
        column_border: int = 3
    ):
        """
        Protected method will calculate the amount of rows depending on plotting type

        Parameters:
            displot_type (bool): Set to True if plotting type is distributional
            column_border (int): The maximum amount of columns to create regardless of plot type

        Returns:
            row_amount (int): Amount of rows to create by plt.subplots
            element_length, column_border (int): Amount of columns to create by plt.subplots
        """
        if len(y_vars) == 0:
            element_length = len(self.x_vars)
        else:
            element_length = len(self.y_vars)

        row_amount = (element_length // column_border) + (element_length % column_border > 0)
        return row_amount, min(element_length, column_border)

    def _initialize_figure_axes (self):
        """
        Will create the figure and the axes using the numbers from _calculate_row_amount for plotting

        Parameters:
            None

        Returns:
            figure (matplotlib.pyplot.figure): The figure to use for plotting
            axes (matplotlib.pyplot.Axes): The axes to use for plotting
        """
        row_amount, column_amount = self._calculate_row_amount()
        figure, axes = matplotlib.pyplot.subplots(nrows=row_amount, ncols=column_amount, figsize=(25.5, 14.5))
        return figure, axes

    def _plot(
        self, 
        axes, 
        plot_function: Callable,
        x_vars: list[str, ...],
        y_vars: list[str, ...],
        dataset: pandas.DataFrame,
        **kwargs
    ):
        """
        Will plot using Seaborn's plotting method while being supplied with arguments

        Parameters:
            axes (matplotlib.pyplot.Axes): The created axes to be used for plotting
            plot_function: (Seaborn's plots): The Seaborn function to use for plotting

        Returns:
            None
        """
        axes = axes.flatten()

        if plot_function in {seaborn.histplot, seaborn.kdeplot, seaborn.ecdfplot, seaborn.countplot}:
            for axes_iteration, column_iteration in enumerate(x_vars):
                plot_function(
                    data=dataset, 
                    x=column_iteration, 
                    ax=axes[axes_iteration], 
                    **kwargs
                )
        else:
            for axes_iteration, column_iteration in enumerate(y_vars):
                plot_function(
                    data=dataset, 
                    x=x_vars, 
                    y=column_iteration, 
                    ax=axes[axes_iteration], 
                    **kwargs
                )

    def plot_relational (
        self,
        relplot_type: str,
        x_vars: list[str, ...],
        y_vars: list[str, ...],
        dataset: pandas.DataFrame
    ):
        figure, axes = self._initialize_figure_axes()
        
        if relplot_type in self.relplot_functions.keys():
            self._plot(axes, self.relplot_functions.get(relplot_type))
        else:
            raise ValueError("[-] Error: User supplied relplot argument doesn't exist")

    def plot_distributional (
        self,
        displot_type: str,
        x_vars: list[str, ...],
        y_vars: list[str, ...],
        dataset: pandas.DataFrame

    ):
        figure, axes = self._initialize_figure_axes()
        if displot_type in self.displot_functions.keys():
            self._plot(axes, self.displot_functions.get(displot_type))
        else:
            raise ValueError("[-] Error: User supplied displot argument doesn't exist")

    def plot_categorical (
        self,
        x_vars: list[str, ...],
        y_vars: list[str, ...],
        dataset: pandas.DataFrame
    ):
        figure, axes = self._initialize_figure_axes()
        if catplot_type in self.catplot_functions.keys():
            self._plot(axes, self.catplot_functions.get(catplot_type))
        else:
            raise ValueError("[-] Error: User supplied catplot argument doesn't exist")
