from typing import *

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.preprocessing import (
    StandardScaler,
    Normalizer,
    MinMaxScaler,
    MaxAbsScaler
)

class ScaleColumns (BaseEstimator, TransformerMixin):
    def __init__ (
        self, 
        columns_to_preprocess : List[str], 
        scaling_preprocessing_type : str, 
        standard_scaler_params : dict, 
        normalizer_params: dict, 
        minmax_params : dict,
        maxabs_params : dict
    ):
        self.columns = columns_to_preprocess
        self.preprocessing_type = scaling_preprocessing_type
        self.standard_scaler_instance = StandardScaler(**standard_scaler_params)
        self.normalizer_instance = Normalizer(**normalizer_params)
        self.minmax_scaler_instance = MinMaxScaler(**minmax_params)
        self.maxabs_scaler_instance = MaxAbsScaler(**maxabs_params)

    def fit_transform (self, X, y=None):
        dataframe_copy = X.copy()

        if self.preprocessing_type == "standard":
            pass
        elif self.preprocessing_type == "normalizer":
            pass
        elif self.preprocessing_type == "minmax":
            pass
        else:
            pass
