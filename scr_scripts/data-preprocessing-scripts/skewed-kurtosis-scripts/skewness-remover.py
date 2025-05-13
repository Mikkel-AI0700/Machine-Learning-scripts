# WARNING: DO NOT IMPORT CLASSES BELOW. USE DEPENDENCY IMPORTER

import logging
from typing import Union, Callable

import numpy
import pandas
import scipy

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RemoveSkew:
    def __init__ (
        self, 
        power_transformer_parameters: dict[str, Any] = None,
        quantile_transformer_parameters: dict[str, Any] = None
    ):
        self.pt_params = power_transformer_parameters
        self.qt_params = quantile_transformer_parameters
        self.mild_negative_threshold = -0.5
        self.mild_positive_threshold = 0.5
        self.severe_negative_threshold = -1
        self.severe_positive_threshold = 1

    def _calculate_skew (
        self,
        columns: Union[list[int, ...], list[str, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if isinstance(dataset, numpy.ndarray):
            return scipy.stats.skew(dataset[:, columns])
        else:
            columns = [columns] if isinstance(columns, str) else columns
            return scipy.stats.skew(dataset[columns])

    def _determine_skew_level (
        self,
        skew_level: Union[int, float]
    ):
        if (skew_level > self.mild_negative_threshold and
            skew_level < self.mild_positive_threshold
        ):
            return "Moderate"
        elif (skew_level > self.severe_negative_threshold or
            skew_level > self.severe_positive_threshold
        ):
            return "Severe"
        else:
            return "Mild"

    def _transform_using_numpy (
        self,
        transformer: Callable,
        columns: list[int, ...],
        dataset: numpy.ndarray
    ):
        dataset_skew_value = self._calculate_skew(columns, dataset)

        if self._determine_skew_level(dataset_skew_value) == "Severe":
            pass
        elif self._determine_skew_level(dataset_skew_value) == "Moderate":
            pass
        else:
            pass

    def _transform_using_pandas (
        self,
        transformer: Callable,
        columns: list[str, ...],
        dataset: pandas.DataFrame
    ):
        dataset_skew_value = self._calculate_skew(columns, dataset)

        if self._determine_skew_level(dataset_skew_value) == "Severe":
            pass
        elif self._determine_skew_level(dataset_skew_value) == "Moderate":
            pass
        else:
            pass
    
    def transform (
        self,
        skew_side: str = None,
        skew_remover: str = None,
        columns: Union[list[int, ...], list[str, ...]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        pass

