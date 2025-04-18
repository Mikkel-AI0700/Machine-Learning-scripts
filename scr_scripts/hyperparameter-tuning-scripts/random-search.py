
# WARNING: DO NOT IMPORT MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

import logging
import numpy
import pandas
from typing import Union, List, Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TrainUsingRandomSearch:
    """
    Automate the RandomizedSearchCV training process and get a dictionary containing all the attributes

    Parameters:
        randomsearch_parameters (Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]]: RandomizedSearchCV parameters
        n_iter (int): Number of iterations for RandomizedSearchCV to look for hyperparameters
    """
    def __init__ (
        self, 
        randomsearch_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]] = None, 
        n_iter: int = 10000
    ):
        self.rs_instance = RandomizedSearchCV(n_iter=n_iter, **randomsearch_parameters)
        self.randomsearch_attributes = {}
        self.model_predictions = {}

    def _distribute_attributes_and_predictions (
        self, 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None, 
        cv_to_pandas: bool = False
    ):
        """
        Assigns all the RandomSearchCV attributes and model predictions in dictionaries

        Parameters:
            test_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Test dataset to use for predictions
            cv_to_pandas (bool): Set to True if cv_results_ attribute is to be converted to Pandas DataFrame

        Returns:
            None
        """
        self.randomsearch_attributes = {
            "best_estimator": self.rs_instance.best_estimator_, 
            "best_score": self.rs_instance.best_score_, 
            "best_params": self.rs_instance.best_params_,
            "scorer_scores": self.rs_instance.scorer_,
            "classes": self.rs_instance.classes_
        }
        self.model_predictions = {
            "base_preds": self.rs_instance.predict(test_dataset_x),
            "proba_preds": self.rs_instance.predict_proba(test_dataset_x),
            "proba_log_preds": self.rs_instance.predict_log_proba(test_dataset_x)
        }

        if cv_to_pandas:
            attributes_to_store.update({"cv_results": pandas.DataFrame(self.rs_instance.cv_results_)})
        else:
            attributes_to_store.update({"cv_results": self.rs_instance.cv_results_})

    def start_randomsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None, 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame] = None, 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        """
        Start the automated RandomizedSearchCV training

        Parameters:
            train_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Independent variables dataset to use to train
            train_dataset_y (Union[numpy.ndarray, pandas.DataFrame]): Dependent variables dataset to use to train
            test_dataset_x (Union[numpy.ndarray, pandas.DataFrame]): Testing dataset to use for _distribute_attributes_and_predictions

        Returns:
            self.randomsearch_attributes (Dict[str, Union[int, float, dict, pandas.DataFrame]): Dictionary containing Randomsearch attributes
            self.model_predictions (Dict[str, numpy.ndarray]): Three unique model predictions: base, proba, log_proba
        """
        if all(dataset != None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]): 
            logging.info("[+] Starting RandomizedSearchCV training")
            self.rs_instance.fit(train_dataset_x, train_dataset_y)

            logging.info("[+] Distributing attributes and model predictions inside dictionaries")
            self._distribute_attributes_and_predictions(test_dataset_x)
            return self.randomsearch_attributes, self.model_predictions
        else:
            raise ValueError("[+] Error: One of the passed datasets is empty. Pass all datasets with values")

    def reset_model_predictions (self, preds_dictionary: Dict[str, numpy.ndarray] = None):
        """
        Method that will reset the original model predictions dictionary
        should it get ruined

        Parameters:
            preds_dictionary (Dict[str, numpy.ndarray]): Original predictions dictionary to reset

        Returns:
            preds_dictionary (Dict[str, numpy.ndarray]): Resetted model predictions dictionary
        """
        logging.info("[*] Resetting RandomizedSearchCV predictions")
        preds_dictionary = self.model_predictions
        return preds_dictionary
