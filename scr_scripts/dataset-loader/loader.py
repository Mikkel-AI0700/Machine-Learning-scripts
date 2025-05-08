
# WARNING: DO NOT IMPORT THE MODULES ABOVE THE CLASS. USE DEPENDENCY IMPORTER

from typing import List, Dict, Union
import os
import re
import logging
import numpy
import pandas
import ucimlrepo

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LoadDataset:
    """
    Dataset loader that will load and create three datasets

    This class will either take in a UCI Machine Learning repository ID or filesystem path
    to load and create three datasets: main dataframe, copy of main dataset, a numpy representation
    of the original dataset

    Parameters:
        uci_id (int): ID of ucimlrepo dataset that will be used to get the dataset
        load_method (str): Option that will get a function reference that will load the dataset
        filesystem_path (str): Filesystem path that points to the dataset file
    """
    def __init__ (self, uci_id: int = None, load_method: str = None, filesystem_path: str = None, **kwargs):
        self.uci_id = uci_id
        self.loader_method = load_method
        self.fs_path = filesystem_path
        self.extra_params = kwargs
        self.datasets = ()
        self.loader_methods = {
            "csv": pandas.read_csv,
            "xlsx": pandas.read_excel,
            "json": pandas.read_json,
            "pickle": pandas.read_pickle,
            "uci": ucimlrepo.fetch_ucirepo
        }

    def _get_loading_method (self):
        """
        Get loader reference if loader_method property exists in loader_methods property keys

        Parameters:
            None

        Returns:
            Pandas dataframe reader or UCI Machine Learning Repository reader
        """
        if self.loader_method in self.loader_methods.keys():
            return self.loader_methods.get(self.loader_method)

    def _create_datasets (self, loaded_dataframe: pandas.DataFrame):
        """
        Function will construct the four datasets that will be returned
        after getting the dataset from _load_pandas or _load_uci

        The four datasets will be saved in an attribute for the reset_datasets
        method

        Parameters 
            loaded_dataset (pandas.DataFrame): The converted dataset from _load_pandas or _load_uci

        Returns:
            tuple (pandas.DataFrame, numpy.ndarray)
        """

        self.datasets = (
            loaded_dataframe,
            loaded_dataframe.copy(),
            loaded_dataframe.to_numpy(),
            loaded_dataframe.to_numpy(copy=True)
        )
        return self.datasets

    def _load_pandas (self):
        """
        Loads a raw dataset from a filesystem path using the Pandas' Input/output functions.
        Will call the self._create_dataset method to construct four datasets

        Parameters:
            None

        Returns:
            datasets (dict): Three datasets, two pandas dataframe and one numpy representation
        """
        loader = self._get_loading_method()
        if os.path.exists(self.fs_path):
            return self._create_datasets(loader(self.fs_path, **self.extra_params))

    def _load_uci (self):
        """
        Will load a temporary raw dataset from UCI Machine Learning Repository, then it will
        assign/create three datasets

        Parameters:
            None

        Returns:
            datasets (dict): Three datasets, two pandas dataframe and one numpy representation
        """
        loader = self._get_loading_method()
        temporary_dataset = loader(id=self.uci_id)
        return self._create_datasets(pandas.DataFrame(temporary_dataset.data.original))

    def load (self, use_uci: bool = False, use_pandas: bool = False):
        """
        Method that allows the user to choose wether to load datasets via Pandas or ucimlrepo

        Parameters:
            use_uci (bool): Set to True if datasets need to be loaded using ucimlrepo
            use_pandas (bool): Set to True if datasets need to be loaded using Pandas

        Returns:
            datasets (dict): Three datasets, two pandas dataframe and one numpy representation
        """
        if use_uci:
            logger.info("[*] Creating three datasets using UCI")
            dataset = self._load_uci()
        if use_pandas:
            logger.info("[*] Creating three datasets using Pandas")
            dataset = self._load_pandas()

        return dataset

    def reset_datasets (self, datasets: tuple[pandas.DataFrame, numpy.ndarray] = None):
        """
        Method will reset the datasets dictionary should the datasets dictionary gets messed up

        Parameters:
            dataset_dict (dict): Dictionary that contains the three original datasets

        Returns:
            dataset_dict (dict): Resetted dictionary
        """
        dataset = self.datasets
        return dataset_dict
