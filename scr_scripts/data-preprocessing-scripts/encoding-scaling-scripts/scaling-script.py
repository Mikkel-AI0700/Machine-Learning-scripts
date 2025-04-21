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

class ScaleColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self,
        columns: Union[str, List[str]] = None,
        scaling_type: str = None,
        scaler_parameters: Dict[str, Any] = None
    ):
        self.columns = columns
        self.scale_type = scaling_type
        self.scaler_params = scaler_parameters
        self.is_numpy = False
        self.is_pandas = False

    def _correct_datatypes (self, dataset: Union[numpy.ndarray, pandas.DataFrame] = None):
        """
        Validates the datatype of the dataset and checks whether it is compatible
        for scaling operations.

        If the dataset is a NumPy array, checks that all elements are either
        integers or floats. If it's a Pandas DataFrame, checks that the specified
        columns contain only numeric datatypes.

        Args:
            dataset (Union[numpy.ndarray, pandas.DataFrame], optional):
                The input dataset to validate. Can be a NumPy array or a Pandas DataFrame.

        Returns:
            bool: True if the dataset has correct datatypes and is compatible with scaling.

        Raises:
            ValueError: If the dataset is not a valid NumPy or Pandas object,
                or if the datatypes within are not numeric.
        """
        try:
            if (isinstance(dataset, numpy.ndarray) and
                all(numpy.issubdtype(dataset, np_type) for np_type in [numpy.integer, numpy.floating])
            ):
                logger.info("Numpy dataset and dataset samples have correct datatypes")
                self.is_numpy = True
                return True
            elif (isinstance(dataset, pandas.DataFrame) and
                all(pandas.api.types.is_numeric_dtype(dataset[col]) for col in self.columns)
            ):
                logger.info("Pandas dataset and dataset samples have correct datatypes")
                self.is_pandas = True
                return True
            else:
                raise ValueError("Either dataset isn't Numpy or Pandas or dataset samples dtypes is incorrect")
        except ValueError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)

    def _transform_dataset (
        self,
        scaler: Callable[Any, Union[numpy.ndarray, pandas.DataFrame]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        """
        Applies the given scaler to the dataset based on the detected format (NumPy or Pandas).

        For NumPy arrays, the entire dataset is scaled. For Pandas DataFrames, only the specified
        columns are scaled. Special handling is applied if the scaler is a Normalizer, which expects
        the entire dataset rather than specific columns.

        Args:
            scaler (Callable):
                A Scikit-Learn compatible scaler instance (e.g., StandardScaler, MinMaxScaler).
            dataset (Union[numpy.ndarray, pandas.DataFrame]):
                The input dataset to be scaled.

        Returns:
            numpy.ndarray or None:
                The scaled dataset if the input was a NumPy array. Returns None for Pandas DataFrames
                as they are modified in-place.
        """
        log_message = "[*] Scaler: {}\n[*] Columns: {}\n[*] Dataset: \n{}\n"

        if self.is_numpy:
            logger.info(log_message.format(scaler.__class__.__name__, dataset, self.columns))
            return scaler.fit_transform(dataset)

        if self.is_pandas:
            if scaler.__class__.__name__ == "Normalizer":
                logger.info(log_message.format("Normalizer", dataset, self.columns))
                dataset[self.columns] = pandas.DataFrame(scaler.fit_transform(dataset))
            else:
                logger.info(log_message.format(scaler.__class__.__name__, dataset, self.columns))
                dataset[self.columns] = pandas.DataFrame(scaler.fit_transform(dataset[self.columns]))

    def fit_transform (self, X, y=None):
        """
        Fits the selected scaler to the input data and transforms it.

        Based on the `scaling_type` provided during initialization, this method
        initializes the corresponding Scikit-Learn scaler, validates the input data,
        and applies the transformation. It supports both NumPy arrays and Pandas DataFrames.

        Args:
            X (Union[numpy.ndarray, pandas.DataFrame]):
                The dataset to be scaled. Must be numeric and either a NumPy array or Pandas DataFrame.
            y (ignored, optional):
                This parameter is included for compatibility with Scikit-Learn's transformer API.

        Returns:
            Union[numpy.ndarray, pandas.DataFrame]:
                The scaled dataset, either as a transformed NumPy array or an updated Pandas DataFrame.

        Logs:
            Error if the scaling type is invalid or the dataset contains incompatible datatypes.
        """
        scaler_instances = {
            "standard": StandardScaler(**(self.scaler_params or {})),
            "minmax": MinMaxScaler(**(self.scaler_params or {})),
            "maxabs": MaxAbsScaler(**(self.scaler_params or {})),
            "Normalizer": Normalizer(**(self.scaler_params or {}))
        }

        if self.scale_type in scaler_instances.keys() and self._correct_datatypes(X):
            self._transform_dataset(scaler_instances.get(self.scale_type), X)
            return X
        else:
            logger.error("Scaling type not in scaling instances or dataset or dataset samples dtype is wrong")

