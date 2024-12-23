import ast
import os

import numpy as np
import pandas as pd
from tools.common.logger import logger

MODEL_MAP = {
    "Crossformer": "Crossformer",
    "DLinear": "DLinear",
    "FEDformer": "FEDformer",
    "FiLM": "FiLM",
    "FITS": "FITS",
    "Informer": "Informer",
    "iTransformer": "iTransformer",
    "MICN": "MICN",
    "NLinear": "NLinear",
    "PatchTST": "PatchTST",
    "RegressionModel": "Linear Regression",
    "RNNModel": "RNN",
    "TCNModel": "TCN",
    "TimeMixer": "TimeMixer",
    "Triformer": "Triformer",
    "VAR_model": "VAR",
    "TimesNet": "TimesNet",
    "Nonstationary_Transformer": "Non-stationary Transformer",
    "Pathformer": "Pathformer",
    "ADARNN": "ADARNN",
    "DUET": "DUET",
    "PDF": "PDF",
}

CSV2DATA_NAME_MAP = {
    "Traffic.csv": "Traffic",
    "Solar.csv": "Solar",
    "ILI.csv": "ILI",
    "Electricity.csv": "Electricity",
    "Weather.csv": "Weather",
    "Exchange.csv": "Exchange",
    "ETTm1.csv": "ETTm1",
    "ETTm2.csv": "ETTm2",
    "ETTh1.csv": "ETTh1",
    "ETTh2.csv": "ETTh2",
    "AQShunyi.csv": "AQShunyi",
    "AQWan.csv": "AQWan",
    "NN5.csv": "NN5",
    "Wike2000.csv": "Wike2000",
    "Wind.csv": "Wind",
    "ZafNoo.csv": "ZafNoo",
    "CzeLan.csv": "CzeLan",
    "Covid-19.csv": "Covid19",
    "NASDAQ.csv": "NASDAQ",
    "NYSE.csv": "NYSE",
    "FRED-MD.csv": "FRED_MD",
    "PEMS04.csv": "PEMS04",
    "PEMS-BAY.csv": "PEMS_BAY",
    "METR-LA.csv": "METR_LA",
    "PEMS08.csv": "PEMS08",

}


def leaderboard2data(detailed_data_name, target_result_name):
    dir_name = os.path.dirname(detailed_data_name)
    base_name = os.path.basename(detailed_data_name)
    save_name = os.path.join(dir_name, 'detail_best_' + base_name)

    detailed_data = pd.read_csv(detailed_data_name)
    target_result = pd.read_csv(target_result_name)
    ret_name = []
    for i in range(0, len(target_result), 2):
        row_pair = target_result.iloc[i:i + 2].reset_index(drop=True)
        info = row_pair["Dataset-Quantity-metrics"][0]
        info = info.split('/')[1].split('-')
        dataset, horizon = info[0], int(info[1])
        for model_name, column_data in row_pair.iloc[:, 1:].items():
            mae_norm, mse_norm = column_data[0], column_data[1]
            if np.isnan(mae_norm) and np.isnan(mse_norm):
                continue
            match_found = False
            for a_index, a_row in detailed_data.iterrows():
                detailed_data_horizon = ast.literal_eval(a_row["strategy_args"].replace('false', 'False'))["horizon"]
                if (isinstance(a_row["file_name"], str) and
                        MODEL_MAP.get(a_row["model_name"], None) == model_name and
                        CSV2DATA_NAME_MAP.get(a_row["file_name"], None) == dataset and
                        round(a_row["mae_norm"], 3) == mae_norm and
                        round(a_row["mse_norm"], 3) == mse_norm and
                        detailed_data_horizon == horizon):
                    indexes = a_row.index
                    ret_name.append(a_row.values)
                    match_found = True
                    break
            if not match_found:
                logger.warning(f"No match found for model '{model_name}', dataset '{dataset}', "
                               f"mae_norm={mae_norm}, mse_norm={mse_norm}")
    res = pd.DataFrame(data=ret_name, columns=indexes)
    res.to_csv(save_name, index=False)
    return save_name


if __name__ == '__main__':
    detailed_data_name = 'res/total.csv'
    target_result_name = "res/best_results_total.csv"
    leaderboard2data(detailed_data_name, target_result_name)
