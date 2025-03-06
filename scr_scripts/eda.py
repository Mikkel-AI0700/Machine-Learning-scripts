from typing import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class InheritorClass:
    def __init__ (
        self, main_dataset : Union[np.ndarray, pd.DataFrame], x_variable : Union[str, List[str]], y_variable : List[str], legend_to_color : str, **kwargs):
        self.main_df = main_dataset
        self.x_variable = x_variable
        self.y_variable = y_variable
        self.hue_class = legend_to_color
        self.extra_kwargs = kwargs

    def _import_required_dependencies (self):
        required_dependencies = ["numpy", "pandas", "seaborn", "matplotlib.pyplot"]
        import importlib


    def _calculate_subplot_amount (self, max_column_border=3):
        temp_column_border = len(max_column_border)
        number_of_rows = (tmep_column_border // max_column_border) + (tmep_column_border % max_column_border > 0)
        return number_of_rows, max_column_border

class RelationalPlots (InheritorClass):
    def __init__ (self):
        self.relplot_function_references = {
            "line" : sns.lineplot,
            "scatter" : sns.scatterplot
        }

    def determine_relational_plot (self, relational_plot_type : str):
        rows, columns = self._calculate_subplot_amount()
        figure, axes = plt.subplots(rows, columns, figsize=(26.5, 15.5))
        axes = axes.flatten()

    def determine_relational_plot (self, relational_plot_type : str):
        if relational_plot_type in self.relplot_function_references.keys():
            relational_plot_reference = self.relplot_function_references.get(relational_plot_type)
            for axes_iteration, column_iteration in enumerate(self.y_variable):
                relational_plot_reference(data=self.main_df, x=self.x_variable, y=column_iteration, hue=self.hue_class, **self.extra_kwargs)

class DistributionalPlots (InheritorClass):
    def __init__ (self):
        self.displot_function_references = {
            "hist" : sns.histplot,
            "kde" : sns.kdeplot,
            "ecdf" : sns.ecdfplot
        }

    def determine_distributional_plot (self, distributional_plot_type : str):
        rows, columns = self._calculate_subplot_amount()
        figure, axes = plt.subplots(rows, columns, figsize=(26.5, 15.5))
        axes = axes.flatten()

        if distributional_plot_type in self.displot_function_references:
            distributional_plot_reference = self.displot_function_references.get(distributional_plot_type, -1)
            for axes_iteration, column_iteration in enumerate(self.y_variable):
                distributional_plot_reference(data=self.main_df, x=self.x_variable, y=column_iteration, hue=self.hue_class, **self.extra_kwargs)

class CategoricalPlots (InheritorClass):
    def __init__ (self):
        self.catplot_function_references = {
            "strip" : sns.stripplot,
            "box" : sns.boxplot,
            "bar" : sns.barplot,
            "count" : sns.countplot
        }

    def determine_categorical_plot (self, categorical_plot_type):
        rows, columns = self._calculate_subplot_amount()
        figure, axes = plt.subplots(rows, columns, figsize=(26.5, 15.5))
        axes = axes.flatten()

        if categorical_plot_type in self.catplot_function_references.keys():
            categorical_plot_reference = self.catplot_function_references.get(categorical_plot_type)
            for axes_iteration, column_iteration in enumerate(self.y_variable):
                categorical_plot_reference(data=self.main_df, x=self.x_variable, y=self.y_variable, hue=self.hue_class)
