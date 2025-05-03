# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import Dict, List, Union
import logging
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    TargetEncoder, 
    LabelEncoder, 
    LabelBinarizer
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EncodeColumns:
    def __init__ (self):
        self.ENCODER_SET = {"ohe", "ordinal", "target", "binarizer", "lencoder"}
        self.TYPE_ERROR_LOG = "[!] Error: NumPy dataset or dataset elements dtype is wrong"
        self.ATTRIBUTE_ERROR_LOG = "[!] Error: Encoder type not in encoder set"

    def _check_types (
        self,
        encoder_type: str,
        columns: Union[int, list[int]],
        dataset: numpy.ndarray
    ):
        try:
            if (not isinstance(dataset, numpy.ndarray) or
                not numpy.issubdtype(dataset[:, columns].dtype, numpy.character)
            ):
                raise TypeError(self.TYPE_ERROR_LOG)
            elif encoder_type not in self.ENCODER_SET:
                raise AttributeError(self.ATTRIBUTE_ERROR_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_encoder_error:
            logger.error(non_existent_encoder_error)

    def _transform_dataset (
        self,
        encoder: Callable,
        columns: Union[int, list[int]],
        dataset: numpy.ndarray
    ):
        if encoder.__class__.__name__ == "OneHotEncoder":
            dataset = numpy.concat(
                (
                    numpy.delete(dataset, columns, axis=1), 
                    encoder.fit_transform(dataset[:, columns])
                ),
                axis=1
            )
        elif encoder.__class__.__name__ == "TargetEncoder":
            temp_x, temp_y = dataset[:, :-1], dataset[:, -1]
            dataset[:, columns] = encoder.fit_transform(temp_x, temp_y)
        else:
            dataset[:, columns] = encoder.fit_transform(dataset[:, columns])

        return dataset

    def transform (
        self,
        encoder_type: str = None,
        encoder_params: dict[str, Any] = None,
        columns: Union[int, list[int]] = None,
        dataset: numpy.ndarray
    ):
        encoder_instances = {
            "ohe": OneHotEncoder(**(encoder_params or None)),
            "ordinal": OrdinalEncoder(**(encoder_params or None)),
            "target": TargetEncoder(**(encoder_params or None)),
            "binarizer": LabelBinarizer(**(encoder_params or None)),
            "lencoder": LabelEncoder(**(encoder_params or None))
        }

        if self._check_types(encoder_type, encoder_instances, columns, dataset):
            dataset = self._transform_dataset(
                encoder_instances.get(encoder_type), columns, dataset
            )
            return dataset

