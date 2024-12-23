import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tools.common.os_contants import RESULT_PATH


def csv2best_detail(
        origin_file_path: str,
        col_name: str,
        save_path: str = "best_detail_result.csv"
) -> str:
    abs_origin_path = os.path.abspath(origin_file_path)
    data = pd.read_csv(abs_origin_path)
    data = data.dropna(subset=['model_name', 'file_name', 'strategy_args', col_name])
    best_result = data.loc[data.groupby(['model_name', 'file_name'])[col_name].idxmax()]

    save_path = os.path.join(RESULT_PATH, save_path)

    folder = os.path.dirname(save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    best_result.to_csv(save_path, index=False)
    return save_path


def csv2best(
        origin_file_path: str,
        col_name: str,
        data_set_names: Optional[List] = None,
        model_names: Optional[List] = None,
        detail_best_save_path: Optional[str] = None,
        best_save_path: Optional[str] = None,
) -> (str, str):
    if best_save_path is None:
        best_save_path = f"best_{col_name}.csv"
    if detail_best_save_path is None:
        detail_best_save_path = f"best_detail_{col_name}.csv"

    best_save_path = os.path.join(RESULT_PATH, best_save_path)
    detail_best_save_path = os.path.join(RESULT_PATH, detail_best_save_path)

    best_detail_result_path = csv2best_detail(origin_file_path, col_name, detail_best_save_path)
    data = pd.read_csv(best_detail_result_path)
    if data_set_names is None:
        data_set_names = data['file_name'].unique()
    if model_names is None:
        model_names = data['model_name'].unique()

    all_combinations = pd.DataFrame(index=data_set_names, columns=model_names)
    filtered_data = data[
        (data['file_name'].isin(data_set_names)) & (data['model_name'].isin(model_names))
        ]

    for (file_name, model_name), group in filtered_data.groupby(['file_name', 'model_name']):
        all_combinations.loc[file_name, model_name] = group[col_name].item()

    folder = os.path.dirname(best_save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    all_combinations.to_csv(best_save_path)
    return best_save_path, best_detail_result_path
