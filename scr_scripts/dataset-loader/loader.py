
# WARNING: DO NOT IMPORT THE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import List, Dict, Union
import os
import re
import logging
import pandas
import ucimlrepo

class LoadDataset:
    def __init__ (self, uci_id: int, load_method: str, filesystem_path: str):
        self.uci_id = uci_id
        self.loader_method = load_method
        self.fs_path = filesystem_path
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
        temp_dictionary = {}

        if os.path.exists(self.fs_path):
            dataset_list = [
                ("main_df", loader(self.fs_path)), 
                ("copy_df", loader(self.fs_path).copy()), 
                ("numpy_df", loader(self.fs_path).to_numpy())
            ]

            for dataset_tuple in dataset_list:
                temp_dictionary.update({dataset_tuple[0]: dataset_tuple[1]})
            return temp_dictionary

    def _load_uci (self):
        loader = self._get_loading_method()
        temp_dictionary = {}
        temporary_dataset = loader(id=self.uci_id)

        dataset_list = [
            ("main_df", pandas.DataFrame(temporary_dataset.data.original)),
            ("copy_df", pandas.DataFrame(temporary_dataset.data.original).copy()),
            ("numpy_df", pandas.DataFrame(temporary_dataset.data.original).to_numpy())
        ]

        for dataset_tuple in dataset_list:
            temp_dictionary.update({dataset_tuple[0]: dataset_tuple[1]})
        return temp_dictionary

    def load (self, load_uci: bool, load_pandas: bool):
        if load_uci:
            dataset_dictionary = self._load_uci()
        if load_pandas:
            dataset_dictionary = self._load_pandas()
        return dataset_dictionary

