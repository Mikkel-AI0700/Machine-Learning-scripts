# WARNING: DO NOT COPY PASTE THE IMPORTS ABOVE THE CLASS. USE DEPENDENCY IMPORTER

import pandas
import logging

logger = getLogger()
logger.setLevel(logging.INFO)

class NullRemover:
    def _check_types (
        self,
        remover_method: str,
        remover_instances: Dict[str, Any],
        columns: Union[str, List[str]],
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
        columns: Union[str, List[str]],
        dataset: pandas.DataFrame
    ):
        pass

    def _iqr_method (
        self,
        columns: Union[str, List[str]],
        dataset: pandas.DataFrame
    ):
        pass

    def transform (
        self,
        remover_method: str,
        columns: Union[str, List[str]],
        dataet: pandas.DataFrame
    ):
        pass

