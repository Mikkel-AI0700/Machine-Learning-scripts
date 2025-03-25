
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import logging
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ScaleColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self, 
        columns_to_preprocess: Union[str, List[str]], 
        scaler_parameters: Dict[str, Union[str, float, numpy.ndarray, pandas.DataFrame]], 
        scaling_preprocessing_type: str, 
        numpy_output: bool, 
        pandas_output: bool
    ):
        self.columns = columns_to_preprocess
        self.scaler_type = scaling_preprocessing_type
        self.scaler_parameters = scaler_parameters
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output

    def _is_correct_datatype (
        self, 
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        numpy_datatypes = (numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.float16, numpy.float32, numpy.float64)
        dataset_datatypes = (numpy.ndarray, pandas.Series, pandas.DataFrame)

        # ----- Two if statements that will check if either numpy or pandas dataframe/series dataset has correct datatypes -----
        if isinstance(dataset, dataset_datatypes[0]) and dataset.dtype in numpy_datatypes:
            return True
        if isinstance(dataset, (dataset_datatypes[1], dataset_datatypes[2])) and dataset[self.columns].dtypes.isin(numpy_datatypes).all():
            return True

    def _transform_dataset (
        self, 
        retain_numpy: bool, 
        retain_pandas: bool, 
        scaler: TransformerMixin, 
        dataset: Union[numpy.ndarray, pandas.Series, pandas.DataFrame]
    ):
        if retain_numpy:
            return scaler.fit_transform(dataset)

        if retain_pandas:
            if scaler.__class__.__name__ == "Normalizer":
                logging.info("[*] Normalizer detected. Doing Normalizer now.")
                dataset = pandas.DataFrame(scaler.fit_transform(dataset.values), dataset.index, dataset.columns)
            else:
                logging.info("[*] {}. Doing {} now".format(scaler.__class__.__name__))
                dataset[self.columns] = pandas.DataFrame(
                    scaler.fit_transform(dataframe_copy[self.columns]), dataset[self.columns].index, dataset[self.columns].columns
                )
            return dataframe_copy

    def fit_transform (self, X, y=None):
        scaler_instances = {
            "standard" : StandardScaler(**(self.scaler_parameters or {})),
            "minmax" : MinMaxScaler(**(self.scaler_parameters or {})),
            "maxabs" : MaxAbsScaler(**(self.scaler_parameters or {})),
            "normalizer" : Normalizer(**(self.scaler_parameters or {}))
        }

        if self.scaler_type in scaler_instances and self._is_correct_datatype(X):
            logging.info("[*] Passing dataset and other parameters to scaler function...")
            transformed_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, scaler_instances.get(self.scaler_type), X)
            return transformed_dataset
        else:
            raise ValueError("[-] Error: Either scaling_preprocessing_type argument doesn't match or dataset type is incorrect")