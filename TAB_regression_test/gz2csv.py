import itertools
import os
import tarfile
from io import StringIO, BytesIO
from typing import Dict, Optional, List

import pandas as pd
from parser import ParserError
from sklearn import logger


def decompress_gz(data: bytes) -> Dict[str, str]:
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                ret[member.name] = tar.extractfile(member).read().decode("utf8")

    return ret


def get_compress_method_from_ext(ext: str) -> Optional[str]:
    return {
        "tar.gz": "gz"
    }.get(ext)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    if method != "gz":
        raise NotImplementedError("Only 'gz' method is supported by now")
    return decompress_gz(data)


def find_key_by_list_string(my_dict, str_list):
    for key, value in my_dict.items():
        if set(str_list).issubset(set(value)):
            return key
    return None


def _find_log_files(directory: str) -> List[str]:
    log_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                log_files.append(os.path.join(root, file))
    return log_files


def read_csvs_from_tar_gz(tar_gz_path):
    dfs = []
    with tarfile.open(tar_gz_path, mode='r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".csv"):
                f = tar.extractfile(member)
                df = pd.read_csv(f)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def read_log_file(fn: str) -> pd.DataFrame:
    return read_csvs_from_tar_gz(fn)


class FieldNames:
    MODEL_NAME = "model_name"
    FILE_NAME = "file_name"
    MODEL_PARAMS = "model_params"
    STRATEGY_ARGS = "strategy_args"
    FIT_TIME = "fit_time"
    INFERENCE_TIME = "inference_time"
    ACTUAL_DATA = "actual_data"
    INFERENCE_DATA = "inference_data"
    LOG_INFO = "log_info"

    @classmethod
    def all_fields(cls) -> List[str]:
        return [
            cls.MODEL_NAME,
            cls.FILE_NAME,
            cls.MODEL_PARAMS,
            cls.STRATEGY_ARGS,
            cls.FIT_TIME,
            cls.INFERENCE_TIME,
            cls.ACTUAL_DATA,
            cls.INFERENCE_DATA,
            cls.LOG_INFO,
        ]


DEFAULT_COLUMNS = [
    FieldNames.MODEL_NAME,
    FieldNames.FILE_NAME,
    FieldNames.MODEL_PARAMS,
    FieldNames.STRATEGY_ARGS,
    FieldNames.FIT_TIME,
    FieldNames.INFERENCE_TIME,
]

ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]


def _load_log_data(log_files: List[str]) -> pd.DataFrame:
    log_files = itertools.chain.from_iterable(
        [[fn] if not os.path.isdir(fn) else _find_log_files(fn) for fn in log_files]
    )
    ret = []
    for fn in log_files:
        logger.info("loading log file %s", fn)
        try:
            res = read_log_file(fn).drop(columns=ARTIFACT_COLUMNS)
            ret.append(res)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # TODO: it is ugly to identify log files by artifact columns...
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def convert_gz_to_csv(folder_path, target_table_columns):
    compressed_files_path = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith("._") and file.endswith(".gz"):
                compressed_files_path.append(file_path)

    files_list_dict = {}
    for file_path in compressed_files_path:
        try:
            col_names = _load_log_data([file_path]).drop(columns=DEFAULT_COLUMNS).columns
            save_file_name = find_key_by_list_string(target_table_columns, col_names)
            if save_file_name is not None:
                files_list_dict.setdefault(save_file_name, [])
                files_list_dict[save_file_name].append(file_path)
            else:
                logger.warning("No matching column names found %s.", file_path)
        except:
            logger.warning("Load %s fail...", file_path)

    data_dict = {}
    for key, items in files_list_dict.items():
        data_dict[key] = _load_log_data(files_list_dict[key])
    return data_dict

