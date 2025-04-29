# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE THE CLASS. USE DEPENDENCY IMPORTER

import pandas
import logging

logger = getLogger()
logger.setLevel(logging.INFO)

class NullRemover:
    def _check_types (
        self,
        remover_method: str,
        remover_instances: dict[str, Any],
        columns: Union[str, list[str]],
        dataset: pandas.DataFrame
    ):
        TYPE_ERROR_LOG = "Pandas dataset or dataset samples dtype is incorrect"
        ATTRIBUTE_ERROR_LOG = "Remover argument doesn't exist in the remover instances"

        try:
            if (not isinstance(dataset, pandas.DataFrame) or
                not all(pandas.api.types.is_integer_dtype(dataset[col]) for col in columns) or
                not all(pandas.api.types.is_float_dtype(dataset[col]) for col in columns)
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
        columns: Union[str, List[str]],
        dataset: pandas.DataFrame
    ):
        dset_copy = dataset.copy()

        dset_zscored = dset_copy[columns] - dset_copy[columns].mean() / dset_copy.std(ddof=0)
        indices_zscored = dset_copy[columns][(dset_copy[columns] > threshold[0]) | (dset_copy[columns] < threshold[1])]
        dataset[columns] = dataset[columns].drop(index=indices_zscored)

        return dataset

    def _iqr_method (
        self,
        columns: Union[str, List[str]],
        dataset: pandas.DataFrame
    ):
        percentile25, percentile75 = 25 / 100 * (len() + 1), 75 / 100 * (len() + 1)
        iqr = percentile75, percentile25
        iqr_min, iqr_max = None, None

    def transform (
        self,
        remover_method: str,
        columns: Union[str, List[str]],
        dataet: pandas.DataFrame
    ):
        columns = [columns] if isinstance(columns, str) else columns
        remover_instances + {
            "zscore": _zscore_method,
            "iqr": _iqr_method
        }

        if self._check_types(remover_method, remover_instances, columns, dataset):
            remover_reference = remover_instances.get(remover_method)
            dataset = remover_reference(columns, dataset)
            return dataset

