import itertools
import os
from parser import ParserError
from typing import List, Dict

import pandas as pd
from sklearn import logger

from tools.utils.tools import read_log_file, _find_log_files, find_key_by_list_string
from tools.common.os_contants import RESULT_PATH


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


# OpenTS 系列中一定存在的字段
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
    """
    加载结果数据。

    如果输入是目录，会递归查找其中的结果文件，并加载其内容。
    加载过程中会过滤掉指定的冗余列，同时跳过无法解析的文件。

    :param log_files: 包含日志文件路径或目录路径的列表。
    :return: 日志数据，合并为一个 DataFrame。
    """
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
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def gz2csv(folder_path: str, target_columns: Dict) -> Dict:
    """
    将指定文件夹中的 .gz 文件解析并转换为按目标列分组的 CSV 数据。

    遍历文件夹及其子文件夹中的所有 .gz 文件，按照目标列名将其分类并加载为 DataFrame。
    无法识别的文件会记录警告日志。

    :param folder_path: 文件夹路径，包含待处理的 .gz 文件。
    :param target_columns: 目标列名的字典，键为存储的文件名，值为对应的列名列表, 会保存所有为文件中包含该列名列表子集的文件。
    :return: 字典，其中键为分类名称，值为加载的 DataFrame。
    """
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
            save_file_name = find_key_by_list_string(target_columns, col_names)
            if save_file_name is not None:
                files_list_dict.setdefault(save_file_name, [])
                files_list_dict[save_file_name].append(file_path)
            else:
                logger.warning("No matching column names found %s.", file_path)
        except:
            logger.warning("Load %s fail...", file_path)

    data_path_dict = {}
    for key, items in files_list_dict.items():
        _load_log_data(files_list_dict[key]).to_csv(RESULT_PATH / key, index=False)
        data_path_dict[key] = RESULT_PATH / key
    return data_path_dict
