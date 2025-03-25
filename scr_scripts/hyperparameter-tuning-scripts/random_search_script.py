
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
        self.rs_global_instance = RandomizedSearchCV(n_iter=n_iter, **standard_randomsearch_parameters)
        self.randomsearch_attributes = {}
        self.model_prediction_results = {}

    def _distribute_attributes_and_predictions (self, test_dataset_x: Union[numpy.ndarray, pandas.DataFrame], convert_cv_to_pandas: bool):
        # ----- Storing the attibrutes into a tuple to be looped -----
        attributes_to_store = [
            ("rs_best_estimator", self.rs_global_instance.best_estimator_), 
            ("rs_best_score", self.rs_global_instance.best_score_), 
            ("rs_best_params", self.rs_global_instance.best_params_),
            ("rs_scorer_scores", self.rs_global_instance.scorer_),
            ("rs_classes", self.rs_global_instance.classes_)
        ]
        model_predictions_to_store = [
            ("model_base_predictions", self.rs_global_instance.predict(test_dataset_x)),
            ("model_proba_predictions", self.rs_global_instance.predict_proba(test_dataset_x))
        ]

        if convert_cv_to_pandas:
            attributes_to_store.append(("rs_cv_results", pandas.DataFrame(self.rs_global_instance.cv_results_)))
        else:
            attributes_to_store.append(("rs_cv_results", self.rs_global_instance.cv_results_))

        # ----- Looping over the attributes_to_store to store into the self.randomsearch_attributes -----
        for attribute_tuple in attributes_to_store:
            self.randomsearch_attributes.update({attribute_tuple[0]: attribute_tuple[1]})

        # ----- Looping over model predictions array to store into self.model_predictions_to_store -----
        for model_predictions_tuple in model_predictions_to_store:
            self.model_prediction_results.update({model_predictions_tuple[0]: model_predictions_tuple[1]})

    def start_randomsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame], 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame], 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if all(dataset == None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]): 
            # ----- Creating instance of RandomizedSearchCV, fitting with self.standard_parameters -----
            print("[+] Starting RandomizedSearchCV training")
            self.rs_global_instance.fit(train_dataset_x, train_dataset_y)

            # ----- Distributing attributes and model predictions in dictionaries -----
            print("[+] Distributing attributes and model predictions into dictionaries...")
            self._distribute_attributes_and_predictions(test_dataset_x)
            return self.randomsearch_attributes, self.model_prediction_results
        else:
            raise ValueError("[+] Error: One of the passed datasets is empty. Pass all datasets with values")

    def reset_model_predictions (self, model_prediction_dictionary: dict):
        logging.info("[*] Resetting RandomizedSearchCV predictions")
        model_prediction_dictionary = self.model_prediction_results
        return model_prediction_dictionary