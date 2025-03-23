
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import numpy
import pandas
from sklearn.model_selection import GridSearchCV

class TrainUsingGridSearch:
    def __init__ (self, standard_gridsearch_parameters: Dict[str, Union[int, float]]):
        self.gs_global_instance = GridSearchCV(**standard_gridsearch_parameters)
        self.gridsearch_attributes = {}
        self.model_predictions = {}

    def _distribute_attributes_and_predictions (self, test_dataset_x: Union[numpy.ndarray, pandas.DataFrame], convert_cv_to_pd: bool):
        # ----- Storing the attributes and predictions into tuples to store into dictionary -----
        attributes_to_store = [
            ("gs_best_estimator", self.gs_global_instance.best_estimator_),
            ("gs_best_score", self.gs_global_instance.best_score_),
            ("gs_best_params", self.gs_global_instance.best_params_),
            ("gs_best_scores", self.gs_global_instance.scorer_)
        ]
        model_predictions_to_store = [
            ("model_base_predictions", self.gs_global_instance.predict(test_dataset_x)),
            ("model_proba_predictions", self.gs_global_instance.predict_proba(test_dataset_x))
        ]

        if convert_cv_to_pd:
            attributes_to_store.append(("gs_cv_results", pandas.DataFrame(self.gs_global_instance.cv_results_)))
        else:
            attributes_to_store.append(("gs_cv_results", self.gs_global_instance.cv_results_))

        # ----- Storing the attributes into tuples to store into dictionary -----
        logging.info("[*] Storing attributes into dictionary to return")
        for attribute_tuple in attributes_to_store:
            self.gridsearch_attributes.update({attribute_tuple[0] : attribute_tuple[1]})

        # ----- Storing the model's predictions into tuples to store into dictionary -----
        logging.info("[+] Storing GridSearchCV model base and probability predictions...")
        for model_preds_tuple in model_predictions_to_store:
            self.model_predictions.update({model_preds_tuple[0] : model_preds_tuple[1]})

    def start_gridsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame], 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame], 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame]
    ):
        logging.basicConfig(level=logging.INFO)

        if all(dataset == None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]):
            # ----- Starting GridSearchCV training -----
            logging.info("[*] Starting GridSearchCV training...")
            self.gs_global_instance.fit(train_dataset_x, train_dataset_y)

            # ----- Distributing the gridsearch attributes into the dictionaries -----
            self._distribute_attributes(gridsearch_instance, test_dataset_x)
            return self.gridsearch_attributes, self.model_predictions
        else:
            raise ValueError("[!] Error: One of the datasets is empty. Pass a ndarray or dataframe dataset") 

    def reset_model_predictions (self, model_prediction_dictionary: Dict[str, Union[numpy.ndarray, pandas.DataFrame]]):
        logging.info("[*] Resetting ML model's predictions")
        model_prediction_dictionary = self.model_predictions
        return model_prediction_dictionary
