# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE THE CLASS. USE DEPENDENCY IMPORTER

import logging
import numpy
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactory

logger = getLogger()
logger.setLevel(logging.INFO)

class NullRemover:
    def __init__ (
        self,
        threshold: tuple[int, int] = None,
        lof_parameters: dict[str, Any] = None,
        iso_parameters: dict[str, Any] = None
    ):
        self.threshold = threshold
        self.lof_params = lof_parameters
        self.iso_params = iso_parameters
        self.REMOVER_INSTANCES = {"zscore", "iqr", "lof", "iso"}
        self.TYPE_ERROR_LOG = "[!] Error: Dataset isn't Numpy or dataset samples are incorrect"
        self.ATTRIBUTE_ERROR_LOG = "[!] Error: Remover method doesn't exist"

    def _check_types (
        self, 
        remover_method: str, 
        columns: Union[int, list[int, int]], 
        dataset: numpy.ndarray
    ):
        try:
            if (not isinstance(dataset, numpy.ndarray) or
                not numpy.issubdtype(dataset[:, columns], numpy.integer) or
                not numpy.issubdtype(dataset[:, columns], numpy.floating)
            ):
                raise TypeError(TYPE_ERROR_LOG)
            elif remover_method not in REMOVER_INSTANCES:
                raise AttributeError(ATTRIBUTE_ERROR_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_remover_error:
            logger.error(non_existent_remover_error)iso_parameters

    def _zscore_method (
        self, 
        threshold: tuple[int, int] = (3, -3),
        columns: Union[int, list[int, int]], 
        dataset: numpy.ndarray
    ):
        dset_cpy = dataset.copy()
        zscored_cpy = dset_cpy[:, columns] - dset_cpy[:, columns].mean() / dset_cpy[:, columns].std(ddof=0)

        dataset[:, columns] = numpy.delete(
            dataset[:, columns], 
            numpy.where((zscored_cpy[:, columns] > threshold[0]) | (zscored_cpy[:, columns] < threshold[1]))
            axis=1
        )

        return dataset

    def _iqr_method (
        self, 
        columns: Union[int, list[int, int]], 
        dataset: numpy.ndarray
    ):
        dset_cpy = dataset[:, columns]

        lower_bound_iqr = (
            numpy.percentile(dset_cpy, 25) - 1.5 * (numpy.percentile(dset_cpy, 75) - numpy.percentile(dset_cpy, 25))
        )
        upper_bound_iqr = (
            numpy.percentile(dset_cpy, 75) + 1.5 * (numpy.percentile(dset_cpy, 75) - numpy.percentile(dset_cpy, 25))
        )

        dataset[:, columns] = numpy.delete(
            dataset[:, columns],
            numpy.where((dataset[:, columns] > upper_bound_iqr) | (dataset[:, columns] < lower_bound_iqr))
            axis=1
        )

        return dataset

    def _lof_method (self, dataset: numpy.ndarray):
        lof_instance = LocalOutlierFactor(**(self.lof_params or None))
        return lof_instance.fit_predict(dataset)

    def _isolation_method (self, dataset: numpy.ndarray):
        iso_instance = IsolationForest(**(self.iso_params or None))
        return iso_instance.fit_predict(dataset)

    def transform (
        self,
        threshold: tuple[int, int] = None,
        remover_method: str = None,
        columns: Union[int, list[int, int]] = None,
        dataset: numpy.ndarray
    ):
        if self._check_types(remover_method, columns, dataset):
            if remover_method is "zscore":
                dataset = self._zscore_method(threshold, columns, dataset)
            elif remover_method is "iqr":
                dataset = self._iqr_method(columns, dataset)
            elif remover_method is "lof":
                dataset = self._lof_method(columns, dataset)
            else:
                dataset = self._iso_method(columns, dataset)
        return dataset

