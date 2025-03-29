
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
    def __init__ (self, standard_randomsearch_parameters: Dict[str, Union[int, float, numpy.ndarray, pandas.DataFrame]], n_iter=10000):
        self.rs_instance = RandomizedSearchCV(n_iter=n_iter, **standard_randomsearch_parameters)
        self.randomsearch_attributes = {}
        self.model_predictions = {}

    def _distribute_attributes_and_predictions (self, test_dataset_x: Union[numpy.ndarray, pandas.DataFrame], convert_cv_to_pandas: bool):
        # ----- Storing the attibrutes into a tuple to be looped -----
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

        if convert_cv_to_pandas:
            attributes_to_store.update({"cv_results": pandas.DataFrame(self.rs_instance.cv_results_)})
        else:
            attributes_to_store.update({"cv_results": self.rs_instance.cv_results_})

    def start_randomsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame], 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame], 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if all(dataset == None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]): 
            # ----- Creating instance of RandomizedSearchCV, fitting with self.standard_parameters -----
            print("[+] Starting RandomizedSearchCV training")
            self.rs_instance.fit(train_dataset_x, train_dataset_y)

            # ----- Distributing attributes and model predictions in dictionaries -----
            print("[+] Distributing attributes and model predictions into dictionaries...")
            self._distribute_attributes_and_predictions(test_dataset_x)
            return self.randomsearch_attributes, self.model_predictions
        else:
            raise ValueError("[+] Error: One of the passed datasets is empty. Pass all datasets with values")

    def reset_model_predictions (self, model_prediction_dictionary: dict):
        logging.info("[*] Resetting RandomizedSearchCV predictions")
        model_prediction_dictionary = self.model_predictions
        return model_prediction_dictionary