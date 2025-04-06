
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
# logging, importlib, typing, and sklearn.base are an exception due to it being a globally required dependency

import logging
import importlib
from typing import List, Dict, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin

# Logger configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ImportRequiredDependencies:
    def import_through_selection (self, standard_module: bool, sklearn_module: bool, package_module: str, modules_to_import: List[str]):
        """
        Dynamically imports modules. Inserts dynamically imported modules inside globals()

        Parameters:
            standard_module (bool): Set to True if importing modules not inside packages
            sklearn_module (bool): Set to True if importing modules from Scikit-Learn
            module (str): Scikit-Learn module to get attribute from
            modules_to_import (Dict[str, str]): Modules containing the key-value modules to import

        Returns:
            None
        """
        try:
            if standard_module:
                for module in modules_to_import:
                    logging.info("[*] Importing module: {}".format(module))
                    globals()[module] = importlib.import_module(module)
            if sklearn_module:
                for module in modules_to_import:
                    logging.info("[*] Importing {}".format(module))
                    globals()[module] = getattr(importlib.import_module(package_module), module)
        except ModuleNotFoundError as non_existent_module:
            logging.error("[!] Error: {}".format(non_existent_module))
            exit(1)

