
# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE CLASS. USE DEPENDENCY IMPORTER
from typing import Dict, List, Union
import logging
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, LabelBinarizer

class EncodeColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self, 
        columns_to_preprocess: Union[str, List[str]], 
        encoder_instance_parameters: Dict[str, Union[str, float, numpy.ndarray, pandas.DataFrame]], 
        encoding_preprocessing_type: str, 
        numpy_output: bool, 
        pandas_output: bool
    ):
        self.columns = columns_to_preprocess
        self.encoding_type = encoding_preprocessing_type
        self.encoder_parameters = encoder_instance_parameters
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output

    def _is_correct_datatype (
        self, 
        check_dataset: bool, 
        check_column: bool, 
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        dataset_datatypes = (numpy.ndarray, pandas.Series, pandas.DataFrame)
        column_datatypes = (numpy.object_, numpy.str_, numpy.int8, "category", "string")

        if check_dataset and isinstance(dataset, dataset_datatypes):
            return True
        else:
            raise ValueError("[-] Error: Dataset argument is not numpy or pandas")

        if check_column:
            if isinstance(dataset, dataset_datatypes[0]) and dataset.dtype in column_datatypes:
                return True
            elif isinstance(dataset, (dataset_datatypes[1], dataset_datatypes[2])) and dataset[self.columns].dtypes.isin(column_datatypes).all():
                return True
            else:
                raise ValueError("[-] Error: Either numpy or pandas dataset's dtype is not correct")

    def _transform_dataset (
        self, 
        retain_numpy: bool, 
        retain_pandas: bool, 
        encoder: TransformerMixin, 
        dataset: Union[numpy.ndarray, pandas.DataFrame]
    ):
        if retain_numpy and self._is_correct_datatype(check_column=True, dataset=dataset) and dataset.ndim > 1:
            logging.info("[*] {} detected. Doing {} now.".format(encoder.__class__.__name__))
            dataset = encoder.fit_transform(dataset)
            return dataset

        if retain_pandas and self._is_correct_datatype(check_column=True, dataset=dataset):
            if encoder.__class__.__name__ == "OneHotEncoder":
                logging.info("[*] OneHotEncoder detected. Doing OneHotEncoding now")
                temp_ohe_dataset = pandas.DataFrame(encoder.fit_transform(dataset[self.columns]), dataset[self.columns].index, dataset[self.columns].columns)
                dataset = pandas.concat([dataset.drop(self.columns, axis=1), temp_ohe_dataset], axis=1)
                return dataset
            else:
                logging.info("[*] {} detected. Doing {} now".format(encoder.__class__.__name__))
                dataset[self.columns] = encoder.fit_transform(dataset[self.columns])
                return dataset

    def fit_transform(self, X, y=None):
        logging.basicConfig(level=logging.INFO)

        encoder_instances = {
            "ohe": OneHotEncoder(**self.encoder_parameters or {}),
            "ordinal": OrdinalEncoder(**self.encoder_parameters or {}),
            "target": TargetEncoder(**self.encoder_parameters or {}),
            "label": LabelEncoder(**self.encoder_parameters or {}),
            "binarizer": LabelBinarizer(**self.encoder_parameters or {})
        }

        if self.encoding_type in encoder_instances.keys() and self._is_correct_datatype(check_dataset=True, dataset=X):
            logging.info("[*] Passing dataset and other parameters now to encoder function...")
            encoded_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, encoder_instances.get(self.encoding_type), X)
            return encoded_dataset
        else:
            raise ValueError("[-] Error: Either encoding argument doesn't exist in the encoder instances or dataset argument contains wrong datatypes")
