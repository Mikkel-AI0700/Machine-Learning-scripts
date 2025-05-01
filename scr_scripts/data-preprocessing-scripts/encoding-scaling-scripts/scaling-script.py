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
    def __init__ (self, scaler_parameters: dict[str, Any] = None):
        self.scaler_params = scaler_parameters
        self.TYPE_ERROR_LOG = "[!] Error: Dataset isn't NumPy or dataset elements have incorrect datatypes"
        self.ATTRIBUTE_ERROR_LOG = "[!] Error: Scaler type argument doesn't exist"
        self.SCALING_INFO_LOG = "[*] Scaler: {}\n[*] Columns: {}\n[*] Dataset: \n{}\n"
        self.scaler_instances = {
            "standard": StandardScaler(**(self.scaler_params or None))
            "minmax": MinMaxScaler(**(self.scaler_params or None)),
            "maxabs": MaxAbsScaler(**(self.scaler_params or None)),
            "normalizer": Normalizer(**(self.scaler_params or None))
        }

    def _check_types (
        self,
        scaler_type: str,
        columns: Union[int, list[int]],
        dataset: numpy.ndarray
    ):
        try:
            if (not isinstance(dataset, numpy.ndarray) or
                not numpy.issubdtype(dataset[:, columns].dtype, numpy.integer) or
                not numpy.issubdtype(dataset[:, columns].dtype, numpy.floating)
            ):
                return TypeError(self.TYPE_ERROR_LOG)
            elif scaler_type not in self.scaler_instances.keys():
                return AttributeError(self.ATTRIBUTE_ERROR_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_scaler_error:
            logger.error(non_existent_scaler_error)

    def _transform_dataset (
        self,
        scaler: Callable
        columns: Union[int, list[int]],
        dataset: numpy.ndarray
    ):
        logger.info(self.SCALING_LOG_INFO.format(scaler, columns, dataset))
        return scaler.fit_transform(dataset[:, columns])

    def transform (
        self,
        scaler_type: str = None,
        columns: Union[int, list[int]] = None,
        dataset: numpy.ndarray = None
    ):
        if self.check_types(scaler_type, columns, dataset):
            dataset = self._transform_dataset(self.scaler_instances.get(scaler_type), columns, dataset)
            return dataset

