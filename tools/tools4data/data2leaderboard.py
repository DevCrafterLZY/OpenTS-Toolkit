import os.path
from pathlib import Path

import pandas as pd

from tools.common.logger import logger
from tools.common.os_contants import STATIC_PATH


def data2leaderboard(file_name: str):
    save_name = file_name.replace('.csv', '_leaderboard.csv')
    multi_df = pd.read_csv(str(file_name))
    # 初始化二维字典
    result_dict = {}
    FILE_NAME_MAP = {
        "FRED-MD": "FRED_MD",
        "PEMS-BAY": "PEMS_BAY",
        "METR-LA": "METR_LA",
        "Covid-19": "Covid19",
    }

    # 遍历multi.csv中的每一行
    for index, row in multi_df.iterrows():
        model_name = row['model_name']
        if model_name == "Nonstationary_Transformer":
            model_name = "Non-stationary Transformer"
        if model_name == "RNNModel":
            model_name = "RNN"
        if model_name == "TCNModel":
            model_name = "TCN"
        if model_name == "VAR_model":
            model_name = "VAR"
        if model_name == "RegressionModel":
            model_name = "Linear Regression"

        strategy_args = eval(row['strategy_args'].replace('false', 'False'))
        horizon = strategy_args['horizon']
        logger.info(row['file_name'])
        if isinstance(row['file_name'], str):
            file_name = row['file_name'].replace('.csv', '')
            mae = round(row['mae_norm'], 3)  # 保留三位小数
            mse = round(row['mse_norm'], 3)  # 保留三位小数

            # 创建字典项
            if model_name not in result_dict:
                result_dict[model_name] = {}
            file_name = FILE_NAME_MAP.get(file_name, file_name)
            result_dict[model_name][f"{file_name}-{horizon}-mae"] = mae
            result_dict[model_name][f"{file_name}-{horizon}-mse"] = mse
    logger.info(result_dict.keys())
    logger.info(result_dict)

    mtsf_df = pd.read_excel(os.path.join(STATIC_PATH, "LEADER_BOARD_TEMPLATE.xlsx"), sheet_name='Sheet1')
    logger.info(mtsf_df)
    for model_name, metrics in result_dict.items():
        for key, value in metrics.items():
            matching_rows = mtsf_df[mtsf_df.apply(lambda row: key in str(row), axis=1)]
            if not matching_rows.empty:
                for idx, row in matching_rows.iterrows():
                    mtsf_df.at[idx, model_name] = value
    logger.info(mtsf_df)
    mtsf_df.to_csv(save_name, index=False)
    logger.info("更新完成，已保存到" + save_name)
    return save_name


if __name__ == '__main__':
    file_name = 'final_12_10/total.csv'
    data2leaderboard(file_name)
