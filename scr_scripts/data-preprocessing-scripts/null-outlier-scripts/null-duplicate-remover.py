
from typing import *
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

class RemoveNull (BaseEstimator, TransformerMixin):
    def __init__ (self, columns_to_preprocess : List[str], removal_type : str, numpy_output : bool, pandas_output : bool, imputer_parameters : dict, null_amount_threshold : int):
        self.columns = columns_to_preprocess
        self.impute_type = removal_type
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output
        self.null_threshold = null_amount_threshold
        self.imputer_params = imputer_parameters
        self.imputer_instances = {"simple" : SimpleImputer, "knn" : KNNImputer}

    def _transform_dataset (
            self, 
            retain_numpy_structure : bool, 
            retain_pandas_structure : bool, 
            drop_nan_columns : bool, 
            drop_duplicates : bool,
            imputer_instance : TransformerMixin, 
            dataset : Union[numpy.ndarray, pandas.Series, pandas.DataFrame]
        ):
        if drop_nan_columns:
            if retain_numpy_structure:
                dataset = dataset[:, ~numpy.isnan(dataset).sum(axis=0) > self.null_threshold]
            if retain_pandas_structure:
                dataset = dataset.loc[:, dataset.isnull().sum() > self.null_threshold]
            return dataset

        if drop_duplicates:
            if drop_duplicates and retain_numpy_structure:
                _, unique_sorted_indices = numpy.unique(dataset, return_index=True)
                dataset = numpy.take(dataset, numpy.sort(unique_sorted_indices))
            if drop_duplicates and retain_pandas_structure:
                dataset[self.columns] = dataset[self.columns].drop_duplicates()
            return dataset

        if imputer_instance.__class__.__name__ != "":
            if retain_numpy_structure and dataset.ndim > 1:
                dataset = imputer_instance.fit_transform(dataset)
            if retain_pandas_structure:
                dataset = pandas.DataFrame(imputer_instance.fit_transform(dataset[self.columns]), dataset[self.columns].columns)
            return dataset

    def fit_transform (self, X, y=None):
        if self.impute_type in self.imputer_instances.keys():
            imputed_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, self.imputer_instances.get(self.impute_type)(self.imputer_params), X)
            return imputed_dataset
        else:
            raise ValueError("[-] Error: Passed imputer type doesn't match any of the existing imputers")
