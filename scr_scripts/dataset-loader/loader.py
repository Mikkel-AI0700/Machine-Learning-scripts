
# WARNING: DO NOT IMPORT THE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import List, Dict, Union
import os
import re
import logging
import pandas
import ucimlrepo

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LoadDataset:
    def __init__ (self, uci_id: int, load_method: str, filesystem_path: str, **kwargs):
        self.uci_id = uci_id
        self.loader_method = load_method
        self.fs_path = filesystem_path
        self.extra_params = kwargs
        self.loader_methods = {
            "csv": pandas.read_csv,
            "xlsx": pandas.read_excel,
            "json": pandas.read_json,
            "pickle": pandas.read_pickle,
            "uci": ucimlrepo.fetch_ucirepo
        }

    def _get_loading_method (self):
        if self.loader_method in self.loader_methods.keys():
            return self.loader_methods.get(self.loader_method)

    def _load_pandas (self):
        loader = self._get_loading_method()

        if os.path.exists(self.fs_path):
            return {
                "main_df": loader(self.fs_path, **self.extra_params),
                "copy_df": loader(self.fs_path, **self.extra_params).copy(),
                "numpy_df": loader(self.fs_path, **self.extra_params).to_numpy()
            }

    def _load_uci (self):
        loader = self._get_loading_method()
        temporary_dataset = loader(id=self.uci_id)

        return {
            "main_df": pandas.DataFrame(temporary_dataset.data.original, **self.extra_params),
            "copy_df": pandas.DataFrame(temporary_dataset.data.original, **self.extra_params).copy(),
            "numpy_df": pandas.DataFrame(temporary_dataset.data.original, **self.extra_params).to_numpy()
        }

    def load (self, use_uci: bool, use_pandas: bool):
        if use_uci:
            logging.info("[*] Creating three datasets using pandas")
            dataset_dictionary = self._load_uci()
        if use_pandas:
            logging.info("[*] Creating three datasets using ucimlrepo")
            dataset_dictionary = self._load_pandas()

        return dataset_dictionary

