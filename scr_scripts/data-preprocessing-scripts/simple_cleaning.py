from typing import *
import importlib

from sklearn.base import *
from sklearn.preprocessing import *

class PreprocessNullValues (BaseEstimator, TransformerMixin):
    def __init__ (self, columns_to_process : List[str]):
        self.columns = columns_to_process

    def _import_required_dependencies (self):
        dependency_list = [
            "sklearn.impute.IterativeImputer", 
            "sklearn.impute.KNNImputer", 
            "sklearn.impute.MissingIndicator", 
            "sklearn.impute.SimpleImputer"
        ]

        for imputer_dependency in dependency_list:
            print("Importing: {}".format(imputer_dependency))
            importlib.import_module(imputer_dependency)

class PreprocessTransformings (BaseEstimator, TransformerMixin):
    
    class ScalerPreprocessing:
        pass

    class 

class PreprocessOutliers (BaseEstimator, TransformerMixin):
    pass

class PreprocessImbalanced (BaseEstimator, TransformerMixin):
    pass
