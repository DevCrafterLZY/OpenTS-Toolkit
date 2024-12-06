import os
from typing import List

import pandas as pd


def find_best_result(origin_file_path: str, col_name: str):
    abs_origin_path = os.path.abspath(origin_file_path)
    data = pd.read_csv(abs_origin_path)

    data = data.dropna(subset=[col_name])
    highest_affiliation_f = data.loc[data.groupby(['model_name', 'file_name'])[col_name].idxmax()]

    origin_dir = os.path.dirname(abs_origin_path)
    save_path = os.path.join(origin_dir, f'best_{col_name}.csv')

    highest_affiliation_f.to_csv(save_path, index=False)
    return save_path


def collect_best_result(data_set_names: List, model_names: List, origin_file_path: str, col_name: str):
    best_result_path = find_best_result(origin_file_path, col_name)
    data = pd.read_csv(best_result_path)

    all_combinations = pd.DataFrame(index=data_set_names, columns=model_names)
    simplified_model_names = [name.split('.')[-1] for name in model_names]
    data['simplified_model_name'] = data['model_name'].apply(lambda x: x.split('.')[-1])

    filtered_data = data[
        (data['file_name'].isin(data_set_names)) & (data['simplified_model_name'].isin(simplified_model_names))
        ]

    for (file_name, simplified_model_name), group in filtered_data.groupby(['file_name', 'simplified_model_name']):
        max_affiliation_f = group[col_name].max()
        full_model_name = next(name for name in model_names if name.endswith(simplified_model_name))
        all_combinations.loc[file_name, full_model_name] = max_affiliation_f
    origin_dir = os.path.dirname(os.path.abspath(origin_file_path))
    save_path = os.path.join(origin_dir, f"all_model_dataset_best_{col_name}_table.csv")
    filtered_save_path = os.path.join(origin_dir, f"filtered_detail_best_{col_name}_table.csv")
    filtered_data.to_csv(filtered_save_path, index=False)
    all_combinations.to_csv(save_path)
    return save_path
