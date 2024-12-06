import os

import pandas as pd

from constant import MODEL_MAP, DATA_NAME2CSV_MAP

metic_map = {
    "F": "affiliation_f",
    "AUC": "auc_roc",
}

type_m_map = {
    "label": "F",
    "score": "AUC",
}


def find_target_row(target: pd.DataFrame, detailed_file: str, metric_name: str) -> str:
    abs_detailed_file_path = os.path.abspath(detailed_file)
    detailed_data_pd = pd.read_csv(abs_detailed_file_path)
    res = []
    for dataset_name, row in target.iterrows():
        for model_name in target.columns:
            metric_val = round(row[model_name], 3)
            for index, detailed_data in detailed_data_pd.iterrows():
                if (isinstance(detailed_data["file_name"], str) and
                        detailed_data["model_name"] == model_name.split('.')[-1] and
                        detailed_data["file_name"] == dataset_name and
                        round(detailed_data[metric_name], 3) == metric_val):
                    res.append(detailed_data.values)
                    print(detailed_data)
                    break
    res = pd.DataFrame(data=res)
    origin_dir = os.path.dirname(abs_detailed_file_path)
    save_path = os.path.join(origin_dir, f'detailed_{metric_name}.csv')
    res.to_csv(save_path, index=False)
    return save_path


if __name__ == '__main__':
    target = pd.read_csv(
        r"E:\CodeProject\LocalProjects\TAB2024_test\result_11_5\all_model_dataset_best_affiliation_f_table.csv",
        index_col=0)
    find_target_row(target, r"E:\CodeProject\LocalProjects\TAB2024_test\result_11_5\best_affiliation_f.csv",
                    "affiliation_f")
