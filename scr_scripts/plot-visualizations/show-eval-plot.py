
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
    Parent class that will serve as the point where child classes will
    get the necessary properties to plot

    Parameters:
        estimator_list (Union[BaseEstimator, List[BaseEstimator]): List of estimators to be plotted in from_estimators
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Params for from_estimators
        predictions_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Params for from_predictions

    Notes:
        It's advised for the user when passing parameters, to pass them inside a dictionary
        It's much easier to modify the arguments when inside a dictionary compared to passing
        it to the class directly
    """
    def __init__ (
        self,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None,
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None,
        predictions_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None
    ):
        self.estimator_list = estimator_list
        self.estimator_params = estimators_parameters
        self.predictions_params = predictions_parameters
        self.figure, self.axes = matplotlib.pyplot.subplots(nrow)

    def _is_estimators (self, estimator_list):
        if all(is_classifier(estimator) for estimator in estimator_list):
            return True

    def _plot (
        self, 
        axes: Axes = None, 
        estimator_list: Union[BaseEstimator, List[BaseEstimator]] = None, 
        metric = None, 
        plot_from_estimators: bool = None, 
        plot_from_predictions: bool = None
    ):
        """
        Method will plot the arguments either from_estimators or from_predictions method

        Parameters:
            axes (matplotlib.pyplot.Axes): The axes to be used for plotting
            estimator_list (Union[BaseEstimator, List[BaseEstimator]]): The list of estimators to plot in from_estimators
            metric: (Callable[Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]], Any of Seaborn's plotting methods]: visualization API plotting method
            plot_from_estimators (bool): Set to True if will plot using from_estimators
            plot_from_predictions (bool): Set to True if will plot using from_predictions

        Returns:
            visualization API plot
        """
        axes = axes.flatten()

        if plot_from_estimators:
            logging.info("[*] Plotting {}".format(metric.__class__.__name__))
            for axes_iteration, estimator_iteration in enumerate(estimator_list):
                metric.from_estimator(estimator=estimator_iteration, ax=axes[axes_iteration], **self.estimator_params)

        if plot_from_predictions:
            logging.info("[*] Plotting {}".format(metric.__class__.__name__))
            metric(**self.predictions_params)

class LearningValidationPlot (InheritorClass):
    """
    Will visualize model from either the LearningCurve or ValidationCurve display

    Parameters:
        plot_type: (str): Parameter to choose if LearningCurveDisplay or ValidationCurveDisplay
        estimator_list (Union[BaseEstimator, List[BaseEstimator]]): List of estimators to plot
        estimators_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]): Arguments for from_estimator method

    """
    def __init__ (
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

    def plot (self):
        if self.plot_type in self.plot_instances.keys() and self._is_estimators(self.estimator_list):
            logging.info("[*] Plot type found, passing necessary parameters to plot function")
            self._plot(self.axes, self.estimator_list, self.plot_instances.get(self.plot_type), plot_from_estimators=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")

class ModelMetricPlots (InheritorClass):
    """
    Visualize the model using Scikit-Learn's Visualization API

    Parameters:
        plot_type (str): The type of plot from the visualization API to use
        estimator_plot (bool): Set to True if will plot using from_estimators
        predictions_plot (bool): Set to True if will plot using from_predictions
        estimator_list
    """
    def __init__ (
        self,
        plot_type: str,
        estimator_plot: bool,
        predictions_plot: bool,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]],
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]],
        predictions_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]
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

    def plot (self):
        if self.plot_type in self.plot_methods.keys() and self._is_estimators(self.estimator_list):
            if self.estimator_plot:
                self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_estimators=True)
            if self.predictions_plot:
                self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_predictions=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")

class InspectionPlots (InheritorClass):
    def __init__ (
        self,
        plot_type: str,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]],
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]
    ):
        super().__init__(estimator_list=estimator_list, estimators_parameters=estimators_parameters)
        self.plot_type = plot_type
        self.plot_methods = {
            "decision": DecisionBoundaryDisplay,
            "partial": PartialDependenceDisplay
        }

    def plot (self):
        if self.plot_type in self.plot_methods.keys() and self._is_estimators(self.estimator_list):
            self._plot(self.axes, self.estimator_list, self.plot_methods.get(self.plot_type), plot_from_estimators=True)
        else:
            raise ValueError("[!] Error: Plot type argument doesn't exist in plot instances or one of the estimators isn't an estimator")
