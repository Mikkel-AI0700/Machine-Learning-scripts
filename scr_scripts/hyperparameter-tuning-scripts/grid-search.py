
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import numpy
import pandas
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TrainUsingGridSearch:
    """
    Automate the GridSearchCV training process and get a dictionary containing all the datasets

    Parameters:
        gridsearch_parameters (Dict[str, Union[int, float]): GridSearchCV parameters
    """
    def __init__ (
        self, 
        gridsearch_parameters: Dict[str, Union[int, float]] = None
    ):
        self.gs_instance = GridSearchCV(**gridsearch_parameters)
        self.gridsearch_attributes = {}
        self.model_predictions = {}

    def _distribute_attributes_and_predictions (
        self, 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None, 
        cv_to_pandas: bool = False
    ):
        """
        Assigns all the attributes and model predictions in dictionaries

        Parameters:
            test_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Test dataset to use for predictions
            cv_to_pandas (bool): Set to True if cv results are to be converted to Pandas DataFrame

        Returns:
            None
        """
        # ----- Storing the attributes and predictions into tuples to store into dictionary -----
        self.gridsearch_attributes = {
            "best_estimator": self.gs_instance.best_estimator_,
            "best_scores": self.gs_instance.best_score_,
            "best_params": self.gs_instance.best_params_,
            "scores": self.gs_instance.scorer_
        }
        self.model_predictions = {
            "base_preds": self.gs_instance.predict(test_dataset_x),
            "proba_preds": self.gs_instance.predict_proba(test_dataset_x),
            "proba_log_preds": self.gs_instance.predict_log_proba(test_dataset_x)
        }

        if cv_to_pandas:
            self.gridsearch_attributes.update({"cv_results": pandas.DataFrame(self.gs_instance.cv_results_)})
        else:
            self.gridsearch_attributes.update({"cv_results": self.gs_instance.cv_results_})

    def start_gridsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None, 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame] = None, 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        """
        Start the automated GridSearchCV training

        Parameters:
            train_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Independent variables dataset to use to train
            train_dataset_y (Union[numpy.ndarray, pandas.DataFrame]): Dependent variables dataset to use to train
            test_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Testing dataset to use for _distribute_attributes_and_predictions

        Returns:
            self.randomsearch_attributes (Dict[str, Union[int, float, dict, pandas.DataFrame]): Dictionary containing Randomsearch attributes
            self.model_predictions (Dict[str, numpy.ndarray]): Three unique model predictions: base, proba, log_proba
        """
        if all(dataset != None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]):
            # ----- Starting GridSearchCV training -----
            logging.info("[*] Starting GridSearchCV training...")
            self.gs_instance.fit(train_dataset_x, train_dataset_y)

            # ----- Distributing the gridsearch attributes into the dictionaries -----
            self._distribute_attributes(gridsearch_instance, test_dataset_x)
            return self.gridsearch_attributes, self.model_predictions
        else:
            raise ValueError("[!] Error: One of the datasets is empty. Pass a ndarray or dataframe dataset") 

    def reset_model_predictions (self, preds_dictionary: Dict[str, Union[numpy.ndarray, pandas.DataFrame]] = None):
        """
        Method that will reset the original model predictions dictionary
        should it get ruined

        Parameters:
            preds_dictionary (Dict[str, numpy.ndarray]): Original predictions dictionary to reset

        Returns:
            preds_dictionary (Dict[str, numpy.ndarray]): Resetted model predictions dictionary
        """
        logging.info("[*] Resetting ML model's predictions")
        preds_dictionary = self.model_predictions
        return preds_dictionary
