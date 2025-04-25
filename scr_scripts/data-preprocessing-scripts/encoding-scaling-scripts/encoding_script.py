# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import Dict, List, Union
import logging
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, LabelBinarizer

logger = logging.getLogger()
logger.setlevel(logging.INFO)

class EncodeColumns:
    def _is_correct_datatype (
        self, 
        columns: Union[str, List[str]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        try:
            if (all(pandas.api.types.is_string_dtype(dataset[col]) for col in columns) or 
                all(pandas.api.types.is_object_dtype(dataset[col]) for col in columns)
            ):
                logger.info("[*] Pandas dataset and samples dtype is correct")
                return True
            else:
                raise TypeError("Pandas dataset or dataset samples dtypes are incorrect")
        except TypeError as incorrect_datatype_error:
            logging.error(incorrect_datatype_error)

    def _transform_dataset (
        self,
        encoder: Callable,
        columns: Union[str, List[str]],
        dataset: Union[numpy.ndarray, pandas.DataFrame],
    ):
        log_message = "[*] Scaler: {}\n[*] Columns: {}\n[*] Dataset: \n{}\n\n"
        logger.info(log_message.format(encoder.__class__.__name__, columns, dataset))

        if encoder.__class__.__name__ == "OneHotEncoder":
            dataset = pandas.concat(
                [dataset.drop(columns, axis=1), pandas.DataFrame(encoder.fit_transform(dataset[columns]))],
                axis=1
            )
        elif encoder.__class__.__name__ == "TargetEncoder":
            temp_x, temp_y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
            dataset = encoder.fit_tranform(temp_x, temp_y)
        else:
            dataset[columns] = encoder.fit_transform(dataset[columns])

    def fit_transform (
        self,
        encoder_type: str = None,
        columns: Union[str, List[str]] = None,
        encoder_parameters: Dict[str, Any] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        encoder_instances = {
            "OneHotEncoder": OneHotEncoder(**(encoder_parameters or {})),
            "OrdinalEncoder": OrdinalEncoder(**(encoder_parameters or {})),
            "TargetEncoder": TargetEncoder(**(encoder_parameters or {})),
            "LabelBinarizer": LabelBinarizer(**(encoder_parameters or {})),
            "LabelEncoder": LabelEncoder(**(encoder_parameters or {}))
        }

        if encoder_type in encoder_instances.keys() and self._is_correct_datatypes(columns, dataset)
            self._transform_dataset(encoder_instances.get(encoder_type), columns, dataset)
        else:
            raise AttributeError("Encoder type argument not in encoder_instances")

