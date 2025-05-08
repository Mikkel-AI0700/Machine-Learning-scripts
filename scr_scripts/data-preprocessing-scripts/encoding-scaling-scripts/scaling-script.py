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
    def _transform (
        self,
        scaler: Callable,
        columns: Union[list[int], list[int, int]],
        dataset: numpy.ndarray
    ):
        dataset[:, columns] = scaler.fit_transform(dataset[:, columns])
        return dataset

    def transform (
        self,
        scaler_type: str = None,
        scaler_params: dict[str, Any] = None,
        columns: Union[list[int], list[int, int]] = None,
        dataset: numpy.ndarray = None
    ):
        scaler_instances = {
            "standard": StandardScaler(**(scaler_params or {})),
            "minmax": MinMaxScaler(**(scaler_params or {})),
            "maxabs": MaxAbsScaler(**(scaler_params or {})),
            "normalizer": Normalizer(**(scaler_params or {}))
        }

        if scaler_type in scaler_instances.keys():
            return self._transform(
                scaler_instances.get(scaler_type), columns, dataset
            )
        else:
            raise AttributeError("[-] Error: User supplied scaler doesn't exist")

