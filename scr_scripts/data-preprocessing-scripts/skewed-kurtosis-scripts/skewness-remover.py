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
        self.pt_instance = PowerTransformer(**(power_transformer_parameters or {}))
        self.qt_instance = QuantileTransformer(**(quantile_transformer_parameters or {}))
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
        if is_numpy:
            return scipy.stats.skew(dataset[:, columns])
        if is_pandas:
            columns = [columns] if isinstance(columns, str) else columns
            return scipy.stats.skew(dataset[columns])

    def _determine_skew_level (
        self,
        skew_values: Union[int, float, list[int, ...], list[float, ...]]
    ):
        skew_level_array = []
        if not isinstance(skew_level_array, list):
            skew_level_array = [skew_level_array]

        for skew_value in skew_values:
            if (skew_value > self.mild_negative_threshold and
                skew_value < self.mild_positive_threshold
            ):
                skew_level_array.append("Moderate")
            elif (skew_value > self.severe_negative_threshold or
                skew_value > self.severe_positive_threshold
            ):
                skew_level_array.append("Severe")
            else:
                skew_level_array.append("Mild")

        return skew_level_array

    def _transform_using_numpy (
        self,
        columns: list[int, ...],
        dataset: numpy.ndarray
    ):
        dataset_skew_value = self._calculate_skew(is_numpy=True, columns=columns, dataset=dataset)
        dataset_skew_levels = self._determine_skew_level(dataset_skew_value)

        for column, skew_level in zip(columns, dataset_skew_levels):
            if skew_level == "Severe":
                dataset[:, [column]] = self.qt_instance.fit_transform(dataset[:, [column]])
            elif skew_level == "Moderate":
                dataset[: [column]] = self.pt_instance.fit_transform(dataset[: [column]])
            else:
                continue

        return dataset

    def _transform_using_pandas (
        self,
        columns: list[str, ...],
        dataset: pandas.DataFrame
    ):
        dataset_skew_value = self._calculate_skew(is_pandas=True, columns=columns, dataset=dataset)
        dataset_skew_levels = self._determine_skew_level(dataset_skew_value)

        for column, skew_level in zip(columns, dataset_skew_levels):
            if skew_level == "Severe":
                dataset[column] = self.qt_instance.fit_transform(dataset[column])
            elif skew_level == "Moderate":
                dataset[column] = self.pt_instance.fit_transform(dataset[column])
            else:
                continue

        return dataset
    
    def transform (
        self,
        columns: Union[list[int, ...], list[str, ...]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        if isinstance(dataset, numpy.ndarray):
            return self._transform_using_numpy(skew_transformers.get(skew_remover), columns, dataset)
        else:
            return self._transform_using_pandas(skew_transformers.get(skew_remover), columns, dataset)
