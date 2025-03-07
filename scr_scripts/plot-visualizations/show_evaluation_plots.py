import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.inspection import *
from sklearn.model_selection import *
from sklearn.base import *
from typing import *

class LearningValidationPlots:
    def __init__ (
        self, 
        estimator_list : Union[BaseEstimator, List[BaseEstimator]], 
        plot_parameters : Dict[str, Union[str, float, Callable]]
    ):
        self.estimator_list = estimator_list
        self.plot_params = plot_parameters
        self.figure, self.axes = plt.subplots(nrows=1, ncols=len(estimator_list), figsize=(15, 15))
        self.required_model_selection_dependencies = [
            "sklearn.model_selection.LearningCurveDisplay",
            "sklearn.model_selection.ValidationCurveDisplay"
        ]

    def _plot_curve (self, learning_curve_plot=False, validation_curve_plot=False):
        self.axes = self.axes.flatten()
        for estimator_index, main_estimator in enumerate(self.estimator_list):
            if learning_curve_plot:
                LearningCurveDisplay.from_estimator(estimator=main_estimator, ax=self.axes[estimator_index], **self.plot_params)
                self.axes[estimator_index].set_title("[+] Learning Curve of {}".format(main_estimator))
                self.axes[estimator_index].grid(True)
            if validation_curve_plot:
                ValidationCurveDisplay.from_estimator(estimator=main_estimator, ax=self.axes[estimator_index], **self.plot_params)
                self.axes[estimator_index].set_title("[+] Validation Curve of {}".format(main_estimator))
                self.axes[estimator_index].grid(True)

    def import_required_dependencies (self):
        try:
            import importlib
            for model_selection_dependency in self.model_selections_dependencies:
                importlib.import_module(model_selection_dependency)
        except ImportError:
            print("[-] Error: One of the required dependencies cannot be imported")

    def learning_curve_from_estimator (self):
        if all(isinstance(estimator, BaseEstimator) for estimator in self.estimator_list):
            self._plot_curve(learning_curve_plot=True)
            plt.title("Learning Curve of {}".format([estimator.split(",") for estimator in self.estimator_list]))
        else:
            print("[-] One of the passed \"estimators\" isn't really an estimator")

    def validation_curve_from_estimator (self):
        if all(isinstance(estimator, BaseEstimator) for estimator in self.estimator_list):
            self._plot_curve(validation_curve_plot=True)
            plt.title("Validation Curve of {}".format([estimator.split(",") for estimator in self.estimator_list]))
        else:
            print("[-] One of the passed \"estimators\" isn't really an estimator")

class ModelMetricPlots:
    def __init__ (
        self, 
        estimator_list : Union[BaseEstimator, List[BaseEstimator]],
        estimator_plot_parameters : Dict[str, Union[str, float, Callable]],
        predictions_plot_parameters : Dict[str, Union[str, float, Callable]]
    ):
        self.estimator_list = estimator_list
        self.from_estimator_parameters = estimator_plot_parameters
        self.figure, self.axes = plt.subplots(nrows=1, ncols=len(estimator_list), figsize=(15, 15))
        self.required_metric_dependencies = [
            "sklearn.metrics.ConfusionMatrixDisplay",
            "sklearn.metrics.RocCurveDisplay",
            "sklearn.metrics.PrecisionRecallDisplay",
            "sklearn.metrics.DetCurveDisplay",
        ]

    def import_required_dependencies (self):
        try:
            import importlib
            for metric_dependency in self.required_metric_dependencies:
                importlib.import_module(metric_dependency)
        except ImportError:
            print("[-] Error: One of the dependencies is missing")

    def confusion_matrix_plot (self):
        self.axes = self.axes.flatten()

        if all(is_classifier(estimator) for estimator in self.estimator_list):
            for classifier_index, main_classifier in enumerate(self.estimator_list):
                ConfusionMatrixDisplay.from_estimator(estimator=main_classifier, ax=self.axes[classifier_index], **self.from_estimator_parameters)
                self.axes[classifier_index].set_title("Confusion Matrix of {}".format(main_classifier))
        else:
            print("[-] Error: One of the passed \"classifiers\" isn't really a classifier")

    def roc_curve_plot (self):
        self.axes = self.axes.flatten()

        if all(is_classifier(estimator) for estimator in self.estimator_list):
            for classifier_index, main_classifier in enumerate(self.estimator_list):
                RocCurveDisplay.from_estimator(estimator=main_classifier, ax=self.axes[classifier_index], **self.from_estimator_parameters)
                self.axes[classifier_index].set_title("ROC Curve plot of {}".format(main_classifier))
        else:
            print("[-] Error: One of the passed \"classifiers\" isn't really a classifier")

    def precision_recall_plot (self):
        self.axes = self.axes.flatten()

        if all(is_classifier(estimator) for estimator in self.estimator_list):
            for classifier_index, main_classifier in enumerate(self.estimator_list):
                PrecisionRecallDisplay.from_estimator(estimator=main_classifier, ax=self.axes[classifier_index], **self.from_estimator_parameters)
                self.axes[classifier_index].set_title("Precision-Recall plot of {}".format(main_classifier))
        else:
            print("[-] Error: One of the passed \"classifiers\" isn't really a classifier")

    def det_plot (self):
        self.axes = self.axes.flatten()
        
        if all(is_classifier(estimator) for estimator in self.estimator_list):
            for classifier_index, main_classifier in enumerate(self.estimator_list):
                DetCurveDisplay.from_estimator(estimator=main_classifier, ax=self.axes[classifier_index], **self.from_estimator_parameters)
                self.axes[classifier_index].set_title("Detection Error Tradeoff plot of {}".format(main_classifier))
        else:
            print("[-] Error: One of the passed \"classifiers\" isn't really a classifier")

class InspectionPlots:
    def __init__ (
        self, 
        estimator_list : Union[BaseEstimator, List[BaseEstimator]],
        estimator_plot_parameters : Dict[str, Union[str, float, Callable]],
        predictions_plot_parameters : Dict[str, Union[str, float, Callable]]
    ):
        self.estimator_list = estimator_list
        self.from_estimator_parameters = estimator_plot_parameters
        self.figure, self.axes = plt.subplots(nrows=1, ncols=len(estimator_list), figsize=(15, 15))
        self.required_inspection_dependencies = [
            "sklearn.inspection.DecisionBoundaryDisplay",
            "sklearn.inspection.PartialDependencyDisplay"
        ]

    def import_required_dependencies (self):
        try:
            import importlib
            for inspection_dependency in self.required_inspection_dependencies:
                importlib.import_module(inspection_dependency)
        except ImportError:
            print("[-] Error: One of the dependencies is missing")

    def decision_boundary_plot (self):
        self._check_dependency_presence()
        self.axes = self.axes.flatten()

        if all(is_classifier(estimator) for estimator in self.estimator_list):
            for classifier_index, main_classifier in enumerate(self.estimator_list):
                DecisionBoundaryDisplay.from_estimator(estimator=main_classifier, ax=self.axes[classifier_index], **self.from_estimator_parameters)
                self.axes[classifier_index].set_title("DecisionBoundary of {}".format(main_classifier))
        else:
            print("[-] One of the passed \"classifier\" is not an classifier")

    def partial_dependence_plot (self):
        self._check_dependency_presence()
        self.axes = self.axes.flatten()

        for estimator_index, main_estimator in enumerate(self.estimator_list):
            PartialDependenceDisplay.from_estimator(estimator=main_estimator, ax=self.axes[estimator_index], **self.from_estimator_parameters)
            self.axes[estimator_index].set_title("PartialDependence of {}".format(main_estimator))
        else:
            print("[-] One of the passed \"classifier\" is not an classifier")
