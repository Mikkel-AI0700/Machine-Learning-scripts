from typing import *

import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

import importlib

class ScaleColumns (BaseEstimator, TransformerMixin):
    def __init__ (self, columns_to_preprocess : List[str], scaling_preprocessing_type : str, scaler_parameters : dict, numpy_output : bool, pandas_output : bool):
        self.columns = columns_to_preprocess
        self.scaler_type = scaling_preprocessing_type
        self.scaler_parameters = scaler_parameters
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output

    def _import_required_dependencies (self):
        required_dependencies = [
            "numpy",
            "pandas",
            "sklearn.base.BaseEstimator",
            "sklearn.base.TransformerMixin",
            "sklearn.preprocessing.StandardScaler",
            "sklearn.preprocessing.Normalizer",
            "sklearn.preprocessing.MinMaxScaler",
            "sklearn.preprocessing.MaxAbsScaler"
        ]

        for dependency in required_dependencies:
            print("[*] Importing: {}".format(dependency))
            importlib.import_module(dependency)

    def _is_correct_datatype (self, dataset_argument : Union[numpy.ndarray, pandas.Series, pandas.DataFrame]):
        numpy_datatypes = [numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.float16, numpy.float32, numpy.float64]

        # ----- Two if statements that will check if either numpy or pandas dataframe/series dataset has correct datatypes -----
        if isinstance(dataset_argument, numpy.ndarray) and dataset_argument.dtype in numpy_datatypes:
            return True
        if isinstance(dataset_argument, (pandas.Series, pandas.DataFrame)) and dataset_argument[self.columns].dtypes.isin(numpy_datatypes).all():
            return True

    def _transform_dataset (self, retain_numpy_structure : bool, retain_pandas_structure : bool, scaler_instance : TransformerMixin, dataset : Union[numpy.ndarray, pandas.Series, pandas.DataFrame]):
            if retain_numpy_structure:
                return scaler_instance.fit_transform(dataset)

            if retain_pandas_structure:
                if scaler_instance.__class__.__name__ == "Normalizer":
                    dataset = pandas.DataFrame(scaler_instance.fit_transform(dataset.values), dataset.index, dataset.columns)
                else:
                    dataset[[self.columns]] = pandas.DataFrame(scaler_instance.fit_transform(dataframe_copy[[self.columns]]), dataset.index, dataset.columns)
                return dataframe_copy

    def fit_transform (self, X, y=None):
        scaler_instances = {
            "standard" : StandardScaler(**(self.scaler_parameters or {})),
            "minmax" : MinMaxScaler(**(self.scaler_parameters or {})),
            "maxabs" : MaxAbsScaler(**(self.scaler_parameters or {})),
            "normalizer" : Normalizer(**(self.scaler_parameters or {}))
        }

        if self.scaler_type in scaler_instances and self._is_correct_datatype(X):
            transformed_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, scaler_instances.get(self.scaler_type), X)
            return transformed_dataset
        else:
            raise ValueError("[-] Error: Either scaling_preprocessing_type argument doesn't match or dataset type is incorrect")