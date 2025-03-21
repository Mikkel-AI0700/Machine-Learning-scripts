
# WARNING: DO NOT COPY PASTE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import List, Union, Dict
import logging
import numpy
import pandas
import seaborn
import matplotlib.pyplot
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.inspection import DecisionBoundaryDisplay, PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, PredictionErrorDisplay, DetCurveDisplay

class InheritorClass:
    def __init__ (
        self,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]],
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]],
        predictions_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]
    ):
        self.estimator_list = estimator_list
        self.estimator_params = estimators_parameters
        self.predictions_params = predictions_parameters
        self.figure, self.axes = matplotlib.pyplot.subplots(nrow)

    def _is_estimators (self, estimator_list):
        if all(is_classifier(estimator) for estimator in estimator_list):
            return True

    def _plot (self, axes, estimator_list, metric_instance, plot_from_estimators, plot_from_predictions):
        axes = axes.flatten()
        if plot_from_estimators:
            for axes_iteration, estimator_iteration in enumerate(estimator_list):
                metric_instance(estimator=estimator_iteration, ax=axes[axes_iteration], **self.estimator_params)

        if plot_from_predictions:
            metric_instance(**self.predictions_params)

class LearningValidationPlot (InheritorClass):
    def __init__ (
        self,
        plot_type: str,
        estimator_list: Union[BaseEstimator, List[BaseEstimator]],
        estimators_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]
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
