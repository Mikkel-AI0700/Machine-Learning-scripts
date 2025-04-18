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
        log_message = "[*] Scaler: {}\n[*] Dataset: \n{}\n[*] Columns: {}"

        if self.is_numpy:
            logger.info(log_message.format(scaler.__class__.__name__, dataset, self.columns))
            return scaler.fit_transform(dataset)

        if self.is_pandas:
            if scaler.__class__.__name__ == "Normalizer":
                logger.info(log_message.format("Normalizer", dataset, self.columns))
                return pandas.DataFrame(scaler.fit_transform(dataset), dataset.columns)
            else:
                logger.info(log_message.format(scaler.__class__.__name__, dataset, self.columns))
                dataset[self.columns] = pandas.DataFrame(
                    scaler.fit_transform(dataset[self.columns]),
                )

    def fit_transform (self, X, y=None):
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
