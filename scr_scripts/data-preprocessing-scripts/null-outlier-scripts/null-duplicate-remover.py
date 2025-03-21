
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

class RemoveNull (BaseEstimator, TransformerMixin):
    def __init__ (self, columns_to_preprocess: List[str], removal_type: str, numpy_output: bool, pandas_output: bool, imputer_parameters: dict, null_amount_threshold: int):
        self.columns = columns_to_preprocess
        self.impute_type = removal_type
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output
        self.null_threshold = null_amount_threshold
        self.imputer_params = imputer_parameters

    def _transform_dataset (
            self, 
            retain_numpy: bool, 
            retain_pandas: bool, 
            drop_nan: bool, 
            drop_duplicates: bool,
            imputer: TransformerMixin, 
            dataset: Union[numpy.ndarray, pandas.Series, pandas.DataFrame]
        ):
        if drop_nan:
            if retain_numpy:
                dataset = dataset[:, ~numpy.isnan(dataset).sum(axis=0) > self.null_threshold]
            if retain_pandas:
                dataset = dataset.loc[:, dataset.isnull().sum() > self.null_threshold]
            return dataset

        if drop_duplicates:
            if drop_duplicates and retain_numpy:
                _, unique_sorted_indices = numpy.unique(dataset, return_index=True)
                dataset = numpy.take(dataset, numpy.sort(unique_sorted_indices))
            if drop_duplicates and retain_pandas:
                dataset[self.columns] = dataset[self.columns].drop_duplicates()
            return dataset

        if imputer.__class__.__name__ != "":
            if retain_numpy and dataset.ndim > 1:
                dataset = imputer.fit_transform(dataset)
            if retain_pandas:
                dataset = pandas.DataFrame(imputer.fit_transform(dataset[self.columns]), dataset[self.columns].columns)
            return dataset

    def fit_transform (self, X, y=None):
        logging.basicConfig(level=logging.INFO)

        imputer_instances = {
            "simple": SimpleImputer(**self.imputer_params or {}),
            "knn": KNNImputer(**self.imputer_params or {})
        }

        if self.impute_type in imputer_instances.keys():
            logging.info("[*] Passing parameters and dataset to function")
            imputed_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, imputer_instances.get(self.impute_type), X)
        else:
            raise ValueError("[-] Error: Passed imputer type doesn't match any of the existing imputers")
