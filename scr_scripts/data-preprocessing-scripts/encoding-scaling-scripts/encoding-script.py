# WARNING: DO NOT COPY AND PASTE THE IMPORTS. USE DEPENDENCY IMPORTER

from typing import Union, Callable
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    LabelBinarizer,
    LabelEncoder
)

class EncodeColumns:
    def __init__ (self):
        self.encoder_instances = {
            "ohe": OneHotEncoder,
            "ordinal": OrdinalEncoder,
            "target": TargetEncoder,
            "binarizer": LabelBinarizer,
            "lencoder": LabelEncoder
        }

    def _transform_using_numpy (
        self,
        encoder_instance: Callable,
        columns: list[int, ...],
        dataset: numpy.ndarray
    ):
        if encoder_instance.__class__.__name__ == "OneHotEncoder":
            return numpy.concat(
                (
                    numpy.delete(dataset, columns, axis=1),
                    encoder_instance.fit_transform(dataset[:, columns])
                ),
                axis=1
            )
        elif encoder_instance.__class__.__name__ == "TargetEncoder":
            train_x, _, train_y, _ = train_test_split(
                dataset[:, :-1],
                dataset[:, -1],
                train_size=0.7,
                test_size=0.3,
                random_state=42,
                shuffle=True
            )
            return encoder_instance.fit_transform(train_x, train_y)
        else:
            return encoder_instance.fit_transform(dataset[:, columns])

    def _transform_using_pandas (
        self,
        encoder_instance: Callable,
        columns: list[str, ...],
        dataset: pandas.DataFrame
    ):
        if encoder_instance.__class__.__name__ == "OneHotEncoder":
            return pandas.concat(
                [
                    dataset.drop(columns, axis=1),
                    encoder_instance.fit_transform(dataset[columns])
                ],
                axis=1
            )
        elif encoder_instance.__class__.__name__ == "TargetEncoder":
            train_x, _, train_y, _ = train_test_split(
                dataset.iloc[:, :-1],
                dataset.iloc[:, -1],
                train_size=0.7,
                test_size=0.3,
                random_state=42,
                shuffle=True
            )
            return encoder_instance.fit_transform(train_x, train_y)
        else:
            return encoder_instance.fit_transform(dataset[columns])

    def transform (
        self,
        encoder_type: str,
        encoder_parameters: dict[str, Any],
        columns: Union[list[str, ...], list[int, ...]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if encoder_type_type not in self.encoder_instances.keys():
            raise ValueError("[-] Error: User supplied encoder doesn't exist")

        if isinstance(dataset, numpy.ndarray):
            return self._transform_using_numpy(
                self.encoder_instances.get(encoder_type)(**(encoder_parameters or {})),
                columns,
                dataset
            )
        if isinstance(dataset, pandas.DataFrame):
            return self._transform_using_pandas(
                self.encoder_instances.get(encoder_type)(**(encoder_parameters or {})),
                columns,
                dataset
            )