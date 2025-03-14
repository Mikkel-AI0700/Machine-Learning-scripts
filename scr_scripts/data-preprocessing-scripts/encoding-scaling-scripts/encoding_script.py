
# IMPORTANT: IMPORT THESE BEFORE RUNNING THE EncodeColumns CLASS!
from typing import *
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, LabelBinarizer

class EncodeColumns (BaseEstimator, TransformerMixin):
    def __init__ (self, columns_to_preprocess : List[str],  encoding_preprocessing_type : str,  encoder_instance_parameters : dict, numpy_output : bool, pandas_output : bool):
        self.columns = columns_to_preprocess
        self.encoding_type = encoding_preprocessing_type
        self.encoder_parameters = encoder_instance_parameters
        self.numpy_output = numpy_output
        self.pandas_output = pandas_output

    def _import_required_dependencies (self):
        import importlib
        required_dependencies = [
            "numpy",
            "pandas",
            "sklearn.base.BaseEstimator",
            "sklearn.base.TransformerMixin",
            "sklearn.preprocessing.OneHotEncoder",
            "sklearn.preprocessing.OrdinalEncoder",
            "sklearn.preprocessing.TargetEncoder",
            "sklearn.preprocessing.LabelEncoder",
            "sklearn.preprocessing.LabelBinarizer"
        ]

        for dependency in required_dependencies:
            print("[*] Importing: {}".format(dependency))
            importlib.import_module(dependency)

    def _is_correct_datatype (self, check_dataset_datatype : bool, check_column_datatype : bool, dataset : Union[numpy.ndarray, pandas.DataFrame]):
        valid_dataset_datatypes = (numpy.ndarray, pandas.Series, pandas.DataFrame)
        valid_column_datatypes = (numpy.object_, numpy.str_, numpy.int8, "category", "string")

        if check_dataset_datatype:
            if isinstance(dataset, valid_dataset_datatypes):
                return True
            else:
                raise ValueError("[-] Error: Dataset argument is not numpy or pandas")

        if check_column_datatype:
            if isinstance(dataset, valid_dataset_datatypes[0]) and dataset.dtype in valid_column_datatypes:
                return True
            elif isinstance(dataset, (valid_dataset_datatypes[1], valid_dataset_datatypes[2])) and dataset[self.columns].dtypes.isin(valid_column_datatypes).all():
                return True
            else:
                raise ValueError("[-] Error: Either numpy or pandas dataset's dtype is not correct")

    def _transform_dataset (self, retain_numpy_structure : bool, retain_pandas_structure : bool, encoder_instance : TransformerMixin, dataset : Union[numpy.ndarray, pandas.Series, pandas.DataFrame]):
        if retain_numpy_structure and dataset.ndim > 1:
            encoded_numpy_dataset = encoder_instance.fit_transform(dataset)
            return encoded_numpy_dataset

        if retain_pandas_structure:
            if encoder_instance.__class__.__name__ == "OneHotEncoder":
                temp_ohe_encoding = pandas.DataFrame(encoder_instance.fit_transform(dataset[self.columns]), dataset[self.columns].index, dataset[self.columns].columns)
                encoded_dataset = pandas.concat([dataset.drop(self.columns, axis=1), temp_ohe_encoding], axis=1)
                return encoded_dataset
            else:
                dataset[self.columns] = encoder_instance.fit_transform(dataset[self.columns])
                return dataset

    def fit_transform(self, X, y=None):
        encoder_instances = {
            "ohe" : OneHotEncoder,
            "ordinal" : OrdinalEncoder,
            "target" : TargetEncoder,
            "label" : LabelEncoder,
            "binarizer" : LabelBinarizer
        }

        self._import_required_dependencies()

        if self.encoding_type in encoder_instances.keys() and self._is_correct_datatype(check_dataset_datatype=True, dataset=X):
            encoded_dataset = self._transform_dataset(self.numpy_output, self.pandas_output, encoder_instances.get(self.encoding_type), X)
            return encoded_dataset
        else:
            raise ValueError("[-] Error: ")
