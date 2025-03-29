
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import numpy
import pandas
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TrainUsingGridSearch:
    def __init__ (self, standard_gridsearch_parameters: Dict[str, Union[int, float]]):
        self.gs_instance = GridSearchCV(**standard_gridsearch_parameters)
        self.gridsearch_attributes = {}
        self.model_predictions = {}

    def _distribute_attributes_and_predictions (self, test_dataset_x: Union[numpy.ndarray, pandas.DataFrame], convert_cv_to_pd: bool):
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

        if convert_cv_to_pd:
            self.gridsearch_attributes.update({"cv_results": pandas.DataFrame(self.gs_instance.cv_results_)})
        else:
            self.gridsearch_attributes.update({"cv_results": self.gs_instance.cv_results_})

    def start_gridsearch_training (
        self, 
        train_dataset_x: Union[numpy.ndarray, pandas.DataFrame], 
        train_dataset_y: Union[numpy.ndarray, pandas.DataFrame], 
        test_dataset_x: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if all(dataset == None for dataset in [train_dataset_x, train_dataset_y, test_dataset_x]):
            # ----- Starting GridSearchCV training -----
            logging.info("[*] Starting GridSearchCV training...")
            self.gs_instance.fit(train_dataset_x, train_dataset_y)

            # ----- Distributing the gridsearch attributes into the dictionaries -----
            self._distribute_attributes(gridsearch_instance, test_dataset_x)
            return self.gridsearch_attributes, self.model_predictions
        else:
            raise ValueError("[!] Error: One of the datasets is empty. Pass a ndarray or dataframe dataset") 

    def reset_model_predictions (self, model_prediction_dictionary: Dict[str, Union[numpy.ndarray, pandas.DataFrame]]):
        logging.info("[*] Resetting ML model's predictions")
        model_prediction_dictionary = self.model_predictions
        return model_prediction_dictionary
