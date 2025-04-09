
# WARNING: DO NOT COPY PASTE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import List, Union, Dict, Callable
import logging
import numpy
import pandas
import seaborn
from matplotlib.pyplot import Axes
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.inspection import DecisionBoundaryDisplay, PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, PredictionErrorDisplay, DetCurveDisplay

class InheritorClass:
    """
    A class that serves as a base for plotting estimators and predictions.

    This class provides the necessary properties and methods to handle estimators,
    their parameters, and predictions for visualization purposes.

    Parameters:
        estimator_list (Union[BaseEstimator, List[BaseEstimator]], optional): A list of estimators to be plotted. Defaults to None.
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]], optional): Parameters for the from_estimators method. Defaults to None.
        predictions_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]], optional): Parameters for the from_predictions method. Defaults to None.

    Notes:
        It is recommended to pass parameters inside a dictionary to make modifications easier later.
    """

    def __init__(
        self,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None,
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None,
        predictions_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None
    ): 
        self.estimator_list = estimator_list
        self.estimator_params = estimators_parameters
        self.predictions_params = predictions_parameters
        self.figure, self.axes = matplotlib.pyplot.subplots(nrows=1)

    def _is_estimators(self, estimator_list):
        """
        Checks if all items in the estimator list are classifiers.

        Args:
            estimator_list (List[BaseEstimator]): A list of estimators to check.

        Returns:
            bool: True if all estimators are classifiers, False otherwise.
        """
        if all(is_classifier(estimator) for estimator in estimator_list):
            return True

    def _plot(
        self,
        axes: Axes = None,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None,
        metric: callable = None,
        plot_from_estimators: bool = None,
        plot_from_predictions: bool = None
    ):
        """
        Plots estimators or predictions using the provided metric.

        Args:
            axes (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
            estimator_list (Union[BaseEstimator, List[BaseEstimator]], optional): The list of estimators to plot. Defaults to None.
            metric (callable, optional): The visualization API method used for plotting. Defaults to None.
            plot_from_estimators (bool, optional): If True, plot using from_estimators. Defaults to None.
            plot_from_predictions (bool, optional): If True, plot using from_predictions. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The generated plot.
        """
        axes = axes.flatten()

        if plot_from_estimators:
            logging.info("[*] Plotting {}".format(metric.__class__.__name__))
            for axes_iteration, estimator_iteration in enumerate(estimator_list):
                metric.from_estimator(estimator=estimator_iteration, ax=axes[axes_iteration], **self.estimator_params)

        if plot_from_predictions:
            logging.info("[*] Plotting {}".format(metric.__class__.__name__))
            metric.from_prediction(**self.predictions_params)


class LearningValidationPlot(InheritorClass):
    """
    Visualizes model performance using either a learning curve or validation curve.

    This class extends `InheritorClass` and provides functionality to generate plots
    from `LearningCurveDisplay` or `ValidationCurveDisplay` using a list of estimators.

    Args:
        plot_type (str, optional): Type of plot to generate. Must be either "learning" or "validation". Defaults to None.
        estimator_list (Union[BaseEstimator, List[BaseEstimator]], optional): A list of estimators to plot. Defaults to None.
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]], optional): Parameters for the from_estimator method. Defaults to None.
    """

    def __init__(
        self,
        plot_type: str = None,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None,
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None
    ):
        super().__init__(estimator_list=estimator_list, estimators_parameters=estimators_parameters)
        self.plot_type = plot_type
        self.plot_instances = {
            "learning": LearningCurveDisplay,
            "validation": ValidationCurveDisplay
        }

    def plot(self):
        """
        Generates the plot using the specified plot type and list of estimators.

        Raises:
            ValueError: If the plot_type is invalid or if any estimator is not a valid classifier.
        """
        if self.plot_type in self.plot_instances.keys() and self._is_estimators(self.estimator_list):
            logging.info("[*] Plot type found, passing necessary parameters to plot function")
            self._plot(self.axes, self.estimator_list, self.plot_instances.get(self.plot_type), plot_from_estimators=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")


class ModelMetricPlots(InheritorClass):
    """Visualize the model using Scikit-Learn's Visualization API.

    Parameters:
        plot_type (str): The type of plot from the visualization API to use.
        estimator_plot (bool): Set to True to plot using from_estimators.
        predictions_plot (bool): Set to True to plot using from_predictions.
        estimator_list (Union[BaseEstimator, List[BaseEstimator]]): The list of estimators to plot.
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Arguments for from_estimator.
        predictions_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Arguments for from_predictions.
    """

    def __init__(
        self, plot_type: str = None, 
        estimator_plot: bool = None, 
        predictions_plot: bool = None, 
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None, 
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None, 
        predictions_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None
    ):
        super().__init__(estimator_list=estimator_list, estimators_parameters=estimators_parameters, predictions_parameters=predictions_parameters)
        self.plot_type = plot_type
        self.estimator_plot = estimator_plot
        self.predictions_plot = predictions_plot
        self.plot_methods = {
            "confusion": ConfusionMatrixDisplay,
            "roc": RocCurveDisplay,
            "precisionrecall": PrecisionRecallDisplay,
            "prediction": PredictionErrorDisplay,
            "det": DetCurveDisplay
        }

    def plot(self):
        """Plots the selected visualization using the configured parameters.

        Raises:
            ValueError: If the plot type is invalid or the estimator list contains invalid entries.
        """
        if self.plot_type in self.plot_methods.keys() and self._is_estimators(self.estimator_list):
            if self.estimator_plot:
                self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_estimators=True)
            if self.predictions_plot:
                self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_predictions=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")

class InspectionPlots(InheritorClass):
    """Visualizes inspection plots such as decision boundaries or partial dependence using configured estimators.

    Parameters:
        plot_type (str): The type of plot to generate. Options include "decision" for decision boundary and "partial" for partial dependence.
        estimator_list (Union[BaseEstimator, List[BaseEstimator]]): A list of estimators to be used for plotting.
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Parameters to be passed to the estimators for plotting.
    """

    def __init__(
        self, plot_type: str = None, 
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None, 
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None
    ):
        super().__init__(estimator_list=estimator_list, estimators_parameters=estimators_parameters)
        self.plot_type = plot_type
        self.plot_methods = {
            "decision": DecisionBoundaryDisplay,
            "partial": PartialDependenceDisplay
        }

    def plot(self):
        """Plots the selected inspection plot using the configured estimators and parameters.

        Raises:
            ValueError: If the plot type is invalid or the estimator list contains invalid entries.
        """
        if self.plot_type in self.plot_methods.keys() and self._is_estimators(self.estimator_list):
            self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_estimators=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")

