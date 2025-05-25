# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import *
import re
import logging
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ScaleColumns:
    def __init__ (self):
        self.scaler_instances = {
            "normalizer": Normalizer,
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "maxabs": MaxAbsScaler
        }

    def _transform_using_numpy (
        self,
        scaler_instance: Callable,
        columns: list[int, ...],
        dataset: numpy.ndarray
    ):
        return scaler_instance.fit_transform(dataset[:, columns])

    def _transform_using_pandas (
        self,
        scaler_instance: Callable,
        columns: list[str, ...],
        dataset: numpy.ndarray
    ):
        return scaler_instance.fit_transform(dataset[columns])

    def transform (
        self,
        scaler_type: str,
        scaler_parameters: dict[str, Any],
        columns: Union[list[int, ...], list[str, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if isinstance(dataset, numpy.ndarray) and scaler_type in self.scaler_instances.keys():
            return self._transform_using_numpy(
                self.scaler_instances.get(scaler_type)(**(scaler_parameters or {})),
                columns,
                dataset
            )
        if isinstance(dataset, pandas.DataFrame) and scaler_type in self.scaler_instances.keys():
            return self._transform_using_pandas(
                self.scaler_instances.get(scaler_type)(**(scaler_parameters or {})),
                columns,
                dataset
            )
