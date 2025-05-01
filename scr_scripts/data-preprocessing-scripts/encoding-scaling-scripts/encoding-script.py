# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import Dict, List, Union
import logging
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, LabelBinarizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EncodeColumns:
    def _check_type_and_existence (
        self,
        encoder_type: str,
        encoder_instances: Dict[str, Callable],
        columns: Union[str, List[str]],
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        col_check = [columns] if isinstance(columns, str) else columns
        TYPE_ERROR_MESSAGE_LOG = "[!] Pandas dataset or dataset dtypes has wrong dtypes"
        ATTRIBUTE_ERROR_MESSAGE_LOG = "[!] Encoder type argument doesn't exist in encoder instances"

        try:
            if (not isinstance(dataset, pandas.DataFrame) or 
                not all(pandas.api.types.is_string_dtype(dataset[col]) for col in col_check)
            ):
                raise TypeError(TYPE_ERROR_MESSAGE_LOG)
            elif encoder_type not in encoder_instances.keys():
                raise AttributeError(ATTRIBUTE_ERROR_MESSAGE_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_encoder_error:
            logger.error(non_existent_encoder_error)

    def _transform_dataset (
        self,
        encoder: Callable,
        columns: Union[str, List[str]],
        dataset: Union[numpy.ndarray, pandas.DataFrame],
    ):
        ENCODER_LOG = "[*] Scaler: {}\n[*] Columns: {}\n[*] Dataset: \n{}\n\n"
        
        logger.info(ENCODER_LOG.format(encoder.__class__.__name__, columns, dataset))
        if encoder.__class__.__name__ == "OneHotEncoder":
            dataset = pandas.concat(
                [dataset.get(columns, axis=1), pandas.DataFrame(encoder.fit_transform(dataset[columns]))],
                axis=1
            )
        elif encoder.__class__.__name__ == "TargetEncoder":
            temp_x, temp_y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
            dataset[columns] = encoder.fit_transform(dataset[columns])
        elif encoder.__class__.__name__ == "OrdinalEncoder":
            dataset[columns] = encoder.fit_transform(dataset[columns])
        else:
            dataset[columns] = encoder.fit_transform(dataset([columns]))
        
        return dataset

    def encode_columns (
        self,
        encoder_type: str = None,
        columns: Union[str, List[str]] = None,
        encoder_parameters: Dict[str, Any] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        encoder_instances = {
            "ohe": OneHotEncoder(**(encoder_parameters or {})),
            "ordinal": OrdinalEncoder(**(encoder_parameters or {})),
            "target": TargetEncoder(**(encoder_parameters or {})),
            "binarizer": LabelBinarizer(**(encoder_parameters or {})),
            "lencoder": LabelEncoder(**(encoder_parameters or {}))
        }

        if self._check_type_and_existence(encoder_type, encoder_instances, columns, dataset):
            self._transform_dataset(encoder_instances.get(encoder_type), columns, dataset)

