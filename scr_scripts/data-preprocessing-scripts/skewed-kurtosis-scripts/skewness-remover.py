# WARNING: DO NOT IMPORT CLASSES BELOW. USE DEPENDENCY IMPORTER

import logging
import numpy
import pandas

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RemoveSkew:
    def __init__ (self):
        pass

    def transform (
        self,
        skew_remover: str = None,
        columns: Union[list[int, ...], list[str, ...]] = None,
        dataset: Union[numpy.ndarray, pandas.DataFrame] = None
    ):
        pass

