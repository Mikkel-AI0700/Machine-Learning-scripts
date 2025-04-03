
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
# importlib and logging IS AN EXCEPTION AS importlib WILL DYNAMICALLY IMPORT ALL THE REQUIRED MODULES AND logging WILL LOG THE MESSAGES
# The typing module imports will remain the same. List, Dict, Union in this class will serve as a universal import
# Including the typing module inside the dictionary dependencies will only pollute it

from typing import List, Dict, Union
import logging
import importlib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ImportRequiredDependencies:
    def __init__ (self):
        self.general_dependencies = {
            "logging": "logging",
            "numpy": "numpy",
            "pandas": "pandas",
            "uci": "ucimlrepo",
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
        self.visual_model_selection = {
            "LearningCurveDisplay": "sklearn.model_selection.LearningCurveDisplay",
            "ValidationCurveDisplay": "sklearn.model_selection.ValidationCurveDisplay"
        }
        self.visual_model_metrics = {
            "ConfusionMatrixDisplay": "sklearn.metrics.ConfusionMatrixDisplay",
            "RocCurveDisplay": "sklearn.metrics.RocCurveDisplay",
            "PrecisionRecallDisplay": "sklearn.metrics.PrecisionRecallDisplay",
            "DetCurveDisplay": "sklearn.metrics.DetCurveDisplay"
        }
        self.visual_model_inspection = {
            "DecisionBoundaryDisplay": "sklearn.inspection.DecisionBoundaryDisplay",
            "PartialDependenceDisplay": "sklearn.inspection.PartialDependenceDisplay"
        }

    def _import_dependencies (self, module_list: List[str], module_dictionary: Dict[str, str]):
        try:
            for (module_name, module_import) in zip(module_list, [module_dictionary.get(module) for module in module_list]):
                logging.info("[*] Imported: {}".format(module_name))
                globals()[module_name] = importlib.import_module(module_import)
        except AttributeError as non_existent_module:
            logging.error("[!] Error: Non existent module: {}".format(non_existent_module))

    def import_through_selection (self, dependency_type: str, required_dependencies: List[str]):
        dependency_map = {
            "general": self.general_dependencies,
            "scaler": self.scaler_dependencies,
            "encoders": self.encoder_dependencies,
            "tuners": self.tuner_dependencies,
            "selection": self.visual_model_selection,
            "metrics": self.visual_model_metrics,
            "inspection": self.visual_model_inspection
        }

        if dependency_type in dependency_map.keys():
            logging.info("[*] Passing the required dependencies to the importer...\n")
            self._import_dependencies(required_dependencies, dependency_map.get(dependency_type))
        else:
            logging.critical("[!] Critical: Argument does not exist in dependency_map keys. Argument: {}".format(dependency_type))
            raise ValueError("[!] Critical: Dependency type argument does not exist in the dictionary map")

