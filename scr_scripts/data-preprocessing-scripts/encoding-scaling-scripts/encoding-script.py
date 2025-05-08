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
    def _transform_dataset (
        self,
        encoder: Callable,
        columns: Union[list[int], list[int, int]],
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
            temp_train_x, _, temp_train_y, _ = train_test_split(
                dataset[:, :-1],
                dataset[:, -1],
                train_size = 0.8,
                test_size = 0.2,
                shuffle = True,
                random_state = 42
            )
            dataset[:, columns] = encoder.fit_transform(temp_x, temp_y)
        else:
            dataset[:, columns] = encoder.fit_transform(dataset[:, columns])

        return dataset

    def transform (
        self,
        encoder_type: str = None,
        encoder_params: dict[str, Any] = None,
        columns: Union[list[int], list[int, int]] = None,
        dataset: numpy.ndarray
    ):
        encoder_instances = {
            "ohe": OneHotEncoder(**(encoder_params or {})),
            "ordinal": OrdinalEncoder(**(encoder_params or {})),
            "target": TargetEncoder(**(encoder_params or {})),
            "binarizer": LabelBinarizer(**(encoder_params or {})),
            "lencoder": LabelEncoder(**(encoder_params or {}))
        }

        if encoder_type in encoder_instances.keys():
            return self._transform_dataset(
                encoder_instances.get(encoder_type), columns, dataset
            )
        else:
            raise AttributeError("[-] Error: User supplied encoder doesn't exist") 

