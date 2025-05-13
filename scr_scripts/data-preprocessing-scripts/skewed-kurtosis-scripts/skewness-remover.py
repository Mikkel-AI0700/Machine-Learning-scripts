# WARNING: DO NOT IMPORT CLASSES BELOW. USE DEPENDENCY IMPORTER

import logging
from typing import Union, Callable

import numpy
import pandas
import scipy
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

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
        self.pt_instance = PowerTransformer(**(self.pt_params))
        self.qt_instance = QuantileTransformer(**(self.qt_params))
        self.mild_negative_threshold = -0.5
        self.mild_positive_threshold = 0.5
        self.severe_negative_threshold = -1
        self.severe_positive_threshold = 1

    def _calculate_skew (
        self,
        is_numpy: bool = False,
        is_pandas: bool = False,
        columns: Union[list[int, ...], list[str, ...]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        if isinstance(dataset, numpy.ndarray):
            return scipy.stats.skew(dataset[:, columns])
        else:
            columns = [columns] if isinstance(columns, str) else columns
            return scipy.stats.skew(dataset[columns])

    def _determine_skew_level (
        self,
        skew_level: Union[list[int, ...], list[float, ...]]
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
        dataset_skew_value = self._calculate_skew(is_numpy=True, columns=columns, dataset=dataset)
        dataset_skew_level = self._determine_skew_level(dataset_skew_value)

        if dataset_skew_level == "Severe":
            return self.qt_instance.fit_transform(dataset[:, columns])
        elif dataset_skew_level == "Moderate":
            return self.pt_instance.fit_transform(dataset[:, columns])
        else:
            return "Mild skew. Not performing any skewness removing"

    def _transform_using_pandas (
        self,
        transformer: Callable,
        columns: list[str, ...],
        dataset: pandas.DataFrame
    ):
        dataset_skew_value = self._calculate_skew(is_pandas=True, columns=columns, dataset=dataset)
        dataset_skew_level = self._determine_skew_level(dataset_skew_value)

        if dataset_skew_level == "Severe":
            return self.qt_instance.fit_transform(dataset[columns])
        elif dataset_skew_level == "Moderate":
            return self.pt_instance.fit_transform(dataset[columns])
        else:
            return "Mild skew. Not performing any skewness removing"
    
    def transform (
        self,
        skew_remover: str = None,
        columns: Union[list[int, ...], list[str, ...]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        if isinstance(dataset, numpy.ndarray):
            self._transform_using_numpy(skew_transformers.get(skew_remover), columns, dataset)
        else:
            self._transform_using_pandas(skew_transformers.get(skew_remover), columns, dataset)
