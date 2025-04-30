# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE THE CLASS. USE DEPENDENCY IMPORTER

import numpy
import logging

logger = getLogger()
logger.setLevel(logging.INFO)

class NullRemover:
    def _check_types (
        self,
        remover_method: str,
        remover_instances: dict[str, Callable],
        columns: Union[int, list[int, int]],
        dataset: numpy.ndarray
    ):
        TYPE_ERROR_LOG = "[!] Error: Dataset isn't Numpy or dataset samples are incorrect"
        ATTRIBUTE_ERROR_LOG = "[!] Error: Remover method doesn't exist"

        try:
            if (not isinstance(dataset, numpy.ndarray) or
                not numpy.issubdtype(dataset[:, columns], numpy.integer) or
                not numpy.issubdtype(dataset[:, columns], numpy.floating)
            ):
                raise TypeError(TYPE_ERROR_LOG)
            elif remover_method not in remover_instances.keys():
                raise AttributeError(ATTRIBUTE_ERROR_LOG)
            else:
                return True
        except TypeError as incorrect_datatype_error:
            logger.error(incorrect_datatype_error)
        except AttributeError as non_existent_remover_error:
            logger.error(non_existent_remover_error)

    def _zscore_method (
        self,
        threshold: tuple[int, int] = (3, -3)
        columns: Union[int, list[int, int]],
        dataset: numpy.ndarray
    ):
        arr_cpy = dataset.copy()

        zscored_cpy = arr_cpy[:, columns] - arr_cpy[:, columns].mean() / arr_cpy[:, columns].std(ddof=0)
        dataset[:, columns] = numpy.delete(
            dataset[:, columns], 
            numpy.where((dataset[:, columns] > threshold[0]) | (dataset[:, columns] < threshold[1]))
            axis=1
        )

        return dataset

    def _iqr_method (
        self,
        columns: Union[int, list[int, int]],
        dataset: numpy.ndarray
    ):
        dset_cols = dataset[:, columns]
        lower_bound_iqr = (
            numpy.percentile(dset_cols, 25) - 1.5 * (numpy.percentile(dset_cols, 75) - numpy.percentile(dset_cols, 25))
        )
        upper_bound_iqr = (
            numpy.percentile(dset_cols, 75) + 1.5 * (numpy.percentile(dset_cols, 75) - numpy.percentile(dset_cols, 25))
        )

        dataset[:, columns] = numpy.delete(
            dataset[:, columns],
            numpy.where((dataset[:, columns] > upper_bound_iqr) | (dataset[:, columns] < lower_bound_iqr))
            axis=1
        )

        return dataset

    def transform (
        self,
        threshold: tuple[int, int] = None,
        remover_method: str = None,
        columns: Union[int, list[int, int]] = None,
        dataset: numpy.ndarray
    ):
        if self._check_types(remover_method, remover_instances, columns, dataset):
            if remover_method is "zscore":
                dataset = self._zscore_method(threshold, columns, dataset)
            else:
                dataset = self._iqr_method(columns, dataset)
        return dataset

