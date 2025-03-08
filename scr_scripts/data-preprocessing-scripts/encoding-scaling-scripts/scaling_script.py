from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler

class ScaleColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self, 
        columns_to_preprocess : List[str] = None, 
        scaling_preprocessing_type : str = None, 
        standard_scaler_params : dict = None, 
        normalizer_params: dict = None, 
        minmax_params : dict = None,
        maxabs_params : dict = None
    ):
        self.columns = columns_to_preprocess
        self.preprocessing_type = scaling_preprocessing_type
        self.standard_scaler_instance = StandardScaler(**(standard_scaler_params or {}))
        self.normalizer_instance = Normalizer(**(normalizer_params or {}))
        self.minmax_scaler_instance = MinMaxScaler(**(minmax_params or {}))
        self.maxabs_scaler_instance = MaxAbsScaler(**(maxabs_params or {}))

    def _is_correct_datatype (self, dataset_argument : Union[np.ndarray, pd.Series, pd.DataFrame]):
        numpy_datatypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]

        # ----- Two if statements that will check if either numpy or pandas dataframe/series dataset has correct datatypes -----
        if isinstance(dataset_argument, np.ndarray) and dataset_argument.dtype in numpy_datatypes:
            return True

        if isinstance(dataset_argument, (pd.Series, pd.DataFrame)) and dataset_argument[self.columns].dtypes.isin(numpy_datatypes).all():
            return True

    def fit_transform (self, X, y=None):
        dataframe_copy = X.copy()
        dataframe_columns, dataframe_index = X.columns, X.index

        # ----- A try-except that has if statements that will do a specific scaling technique -----
        # ----- Will assign the scaled column features by converting it first to a dataframe then assign it to dataframe -----
        try:
            if self.preprocessing_type == "standard" and self._is_correct_datatype(dataframe_copy):
                dataframe_copy[self.columns] = pd.DataFrame(
                    data=self.standard_scaler_instance.fit_transform(dataframe_copy[self.columns]), 
                    index=dataframe_index,
                    columns=dataframe_columns
                )
                return dataframe_copy
            elif self.preprocessing_type == "minmax" and self._is_correct_datatype(dataframe_copy):
                dataframe_copy[self.columns] = pd.DataFrame(
                    data=self.minmax_scaler_instance(dataframe_copy[self.columns]),
                    index=dataframe_index,
                    columns=dataframe_columns
                )
                return dataframe_copy
            elif self.preprocessing_type == "maxabs" and self._is_correct_datatype(dataframe_copy):
                dataframe_copy[self.columns] = pd.DataFrame(
                    data=self.maxabs_scaler_instance(dataframe_copy[self.columns]),
                    index=dataframe_index,
                    columns=dataframe_columns
                )
                return dataframe_copy
            elif self.preprocessing_type == "normalizer" and self._is_correct_datatype(dataframe_copy):
                pass
            else:
                raise ValueError("Incorrect scaling type or incorrect dataset type passed as argument")
        except ValueError as incorrect_arguments:
            print("[-] Error: {}".format(incorrect_arguments))
            exit(1)
