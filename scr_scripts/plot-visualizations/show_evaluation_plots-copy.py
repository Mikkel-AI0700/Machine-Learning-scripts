
# WARNING: DO NOT COPY PASTE MODULES ABOVE THE CLASSES. USE DEPENDENCY IMPORTER

from typing import *
import logging
import numpy
import pandas
import seaborn
import matplotlib.pyplot
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, DetCurveDisplay

class InheritorClass:
    def __init__ (
        self, 
        estimator_list: Union[BaseEstimator, List[BaseEstimator],
        estimator_plot_parameters: Dict[str, Union[str, float]],
        predictions_plot_parameters: Dict[str, Union[str, float]]
    ):
        self.estimator_list = estimator_list
        self.estimator_parameters = estimator_plot_parameters
        self.prediction_parameters = predictions_plot_parameters
        self.figure, self.axes = matplotlib.pyplot.subplots(nrows=1, ncols=len(estimator_list), figsize=(26.5, 15.5))

    def _is_estimators (self, estimator_list: Union[BaseEstimator, List[BaseEstimator]]):
        if all(is_classifier(estimator) for estimator in estimator_list):
            return True

    def _plot (self, axes: matplotlib.pyplot.Axes, metric_instance):
        for axes_iteration, estimator_iteration in enumerate(self.estimator_list):
            metric_instance()


class LearningValidationPlots (InheritorClass):
    def __init__ (
        self,
        estimator_list: Union[BaseEstimator, List[BaseEstimator],
        estimator_plot_parameters: Dict[str, Union[str, float, Callable]],
        predictions_plot_parameters: Dict[str, Union[str, float]]
    ):
        super().__init__(estimator_list, estimator_plot_parameters, predictions_plot)
        self.plot_selections = {
            "learning": LearningCurveDisplay,
            "validation": ValidationCurveDisplay
        }

    def determine_plot (self, plot_type: str):
        if plot_type in self.plot_selections.keys() and self._is_estimators(self.estimator_list):
            logging.info("[*] Passing axes and display type to function")
            self.axes = self.axes.flatten()
            self._plot(self.axes, self.plot_selections(plot_type))
        else:
            raise ValueError("[!] Error: Either the plot type doesn't exist or one of the estimators in list isn't an argument")

class ModelMetricPlots (InheritorClass):
    def __init__ (
        self,
        estimator_list: Union[BaseEstimator, List[BaseEstimator],
        estimator_plot_parameters: Dict[str, Union[str, float, Callable]],
        predictions_plot_parameters: Dict[str, Union[str, float]]
    ):
        super().__init__(estimator_list, estimator_plot_parameters, predictions_plot_parameters)
        self.plot_selections = {
            "confusion": ConfusionMatrix,
            "roc": RocCurveDisplay,
            "precisionrecall": PrecisionRecallDisplay,
            "DetCurveDisplay": DetCurveDisplay
        }

    def determine_plot (self, plot_type: str):
        if plot_type in self.plot_selections.keys() and self._is_estimators(self.estimator_list):
            logging.info()

class InspectionPlots (InheritorClass):
    def __init__ (self):
        pass

