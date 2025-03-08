from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, LabelBinarizer

class EncodeColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self,
        columns_to_preprocess : List[str] = None, 
        encoding_preprocessing_type : str = None, 
    ):
        self.columns = columns_to_preprocess
        self.encoding_type = encoding_preprocessing_type

    def _is_correct_datatype (self, dataset_argument : Union[np.ndarray, pd.DataFrame]):
        pass

    def fit_transform(self, X, y=None):
        dataframe_copy, dataframe_columns, dataframe_index = X.copy, X.columns, X.index

        try:
            pass
        except ValueError as incorrect_arguments:
            print("[-] Error: {}".format(incorrect_arguments))
            exit(1)
