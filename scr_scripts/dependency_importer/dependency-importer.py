
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import importlib

class ImportRequiredDependencies:
    def __init__ (self):
        self.general_dependencies = {
            "logging": "logging",
            "numpy": "numpy",
            "pandas": "pandas",
            "seaborn": "seaborn",
            "matplotlib": "matplotlib.pyplot"
        }
        self.scaler_dependencies = {
            "Normalizer": "sklearn.preprocessing.Normalizer",
            "StandardScaler": "sklearn.preprocessing.StandardScaler",
            "MinMaxScaler": "sklearn.preprocessing.MinMaxScaler",
            "MaxAbsScaler": "sklearn.preprocessing.MaxAbsScaler"
        }
        self.encoder_dependencies = {
            "OneHotEncoder": "sklearn.preprocessing.OneHotEncoder",
            "OrdinalEncoder": "sklearn.preprocessing.OrdinalEncoder",
            "TargetEncoder": "sklearn.preprocessing.TargetEncoder",
            "LabelEncoder": "sklearn.preprocessing.LabelEncoder",
            "LabelBinarizer": "sklearn.preprocessing.LabelBinarizer",
        }
        self.tuner_dependencies = {
            "GridSearch": "sklearn.model_selection.GridSearchCV",
            "RandomizedSearch": "sklearn.model_selection.RandomizedSearchCV"
        }

    def _import_dependencies (self, dependency_list: List[str], dependency_dictionary: dict):
        try:
            for dependency in [dependency_dictionary.get(dependency) for dependency in dependency_list]:
                logging.info("[*] Importing: {}".format(dependency))
                importlib.import_module(dependency)
        except ImportError as non_existent_module:
            logging.warn("[!] Failed to import: {} | {}".format(dependency, non_existent_module))

    def import_through_selection (self, dependency_type: str, required_dependencies: List[str]):
        dependency_map = {
            "general": self.general_dependencies,
            "scaler": self.scaler_dependencies,
            "encoders": self.encoder_dependencies,
            "tuners": self.tuner_dependencies
        }

        if dependency_type in dependency_map.keys():
            logging.info("[*] Passing the required dependencies to the importer...")
            self._import_dependencies(required_dependencies, dependency_map.get(dependency_type))
        else:
            logging.critical("[!] Critical: Argument does not exist in dependency_map keys. Argument: {}".format(dependency_type))
            raise ValueError("[-] Critical: Dependency type argument does not exist in the dictionary map")

