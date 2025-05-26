# WARNING: DO NOT IMPORT CLASSES BELOW. USE DEPENDENCY IMPORTER

import logging
from typing import Union, Callable, Any

import numpy
import pandas
import scipy
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RemoveSkew:
    def __init__ (
        self,
        power_transformer_params: dict[str, Any] = None,
        quantile_transformer_params: dict[str, Any] = None
    ):
        self.is_numpy = False,
        self.is_pandas = False

    def _fix_skew (
        self,
        skew_method: str,
        remover_function: Callable,
        columns: Union[list[str, ...], list[int, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if self.is_numpy:
            return remover_function(dataset[:, columns])

        if self.is_pandas:
            return remover_function(dataset[columns])

    def calculate_skew (
        self,
        columns: Union[list[int, ...], list[str, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if isinstance(dataset, numpy.ndarray):
            return scipy.stats.skew(dataset[:, columns])
        if isinstance(dataset, pandas.DataFrame):
            return scipy.stats.skew(dataset[columns])

    def transform (
        self,
        skew_method: str,
        skew_func_params: dict[str, Any],
        columns: Union[list[str, ...], list[int, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        pass
