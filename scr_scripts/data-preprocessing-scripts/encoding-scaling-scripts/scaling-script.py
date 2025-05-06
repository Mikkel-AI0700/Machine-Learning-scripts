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
        self.SCALER_SET = {"standard", "minmax", "maxabs", "normalizer"}
        self.TYPE_ERROR_LOG = "[-] Error: Dataset isn't numpy or dataset samples dtype incorrect"
        self.ATTRIBUTE_ERROR_LOG = "[-] Error: Scaler argument not in scaler set"

    def _check_types (
        self,
        scaler_type: str,
        columns: Union[list[int], list[int, int]],
        dataset: numpy.ndarray
    ):
        try:
            if (not isinstance(dataset, numpy.ndarray) 
                #not numpy.issubdtype(dataset[:, columns].dtype, numpy.integer) and
                #not numpy.issubdtype(dataset[:, columns].dtype, numpy.floating)
            ):
                raise TypeError(self.TYPE_ERROR_LOG)
            elif scaler_type not in self.SCALER_SET:
                raise AttributeError(self.ATTRIBUTE_ERROR_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_scaler_error:
            logger.error(non_existent_scaler_error)

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

        if self._check_types(scaler_type, columns, dataset):
            return self._transform(
                scaler_instances.get(scaler_type), columns, dataset
            )

