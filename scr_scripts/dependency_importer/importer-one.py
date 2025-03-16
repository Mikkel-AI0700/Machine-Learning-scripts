
from typing import *
import importlib

class ImportDependencies:
	def __init__ (self):
		self.general_dependencies = {
			"numpy" : "numpy",
			"pandas" : "pandas",
			"seaborn" : "seaborn",
			"matplotlib" : "matplotlib.pyplot",
		}
		self.scaler_dependencies = {
			"Normalizer" : "sklearn.preprocessing.Normalizer",
			"StandardScaler" : "sklearn.preprocessing.StandardScaler",
			"MinMaxScaler" : "sklearn.preprocessing.MinMaxScaler",
			"MaxAbsScaler" : "sklearn.preprocessing.MaxAbsScaler"
		}
		self.encoder_dependencies = {
			"OneHotEncoder" : "sklearn.preprocessing.OneHotEncoder",
			"OrdinalEncoder" : "sklearn.preprocessing.OrdinalEncoder",
			"TargetEncoder" : "sklearn.preprocessing.TargetEncoder",
			"LabelEncoder" : "sklearn.preprocessing.LabelEncoder"
			"LabelBinarizer" : "sklearn.preprocessing.LabelBinarizer"
		}

	def _import_dependencies (self, dependency_list : List[str], dependency_dictionary : dict):
		pass

	def determine_dependency (self, dependency_type : str, required_dependencies : List[str]):
		if dependency_type == "scalers" and required_dependencies in self.scaler_dependencies:
			pass
		elif dependency_type == "encoders" and required_dependencies in self.encoder_dependencies:
			pass

