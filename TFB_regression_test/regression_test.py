import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from gz2csv import convert_gz_to_csv
from ts_benchmark.common.constant import ROOT_PATH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(CURRENT_DIR, "example")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def show_diff(example_data, data2):
    example_data = example_data.sort_values(by=['model_name', 'file_name'])
    data2 = data2.sort_values(by=['model_name', 'file_name'])
    merged_data = pd.merge(example_data, data2, on=['model_name', 'file_name'], suffixes=('_1', '_2'), how='left')

    difference = pd.DataFrame({'model_name': merged_data['model_name'], 'file_name': merged_data['file_name']})

    for column in example_data.columns:
        if column == 'model_name' or column == 'file_name':
            continue

        if merged_data[column + '_1'].dtype == 'object':
            difference[column] = merged_data[column + '_1'] == merged_data[column + '_2']
        else:
            difference[column] = merged_data[column + '_1'] - merged_data[column + '_2']

    print("Comparison completed. Differences:")
    print(difference)
    return difference


python_path = sys.executable
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_results_path = f"regression_test/{time_stamp}"
model_results_absolute_path = os.path.join(ROOT_PATH, "result", model_results_path)
val_diff_result_path = f"val_result/{time_stamp}"

test_sh = f"{EXAMPLE_DIR}/test.sh"
subprocess.run(["bash", test_sh, python_path, ROOT_PATH, model_results_path])

log_data = convert_gz_to_csv(model_results_absolute_path)
os.makedirs(val_diff_result_path, exist_ok=True)
log_data.to_csv(f"{CURRENT_DIR}/{val_diff_result_path}/{time_stamp}.csv", index=False)

example = pd.read_csv(f"{EXAMPLE_DIR}/example_output.csv")
diff = show_diff(example, log_data)
diff.to_csv(f"{val_diff_result_path}/{time_stamp}_diff.csv", index=False)
print(f"Saving new results to CSV: {val_diff_result_path}/{time_stamp}.csv")
print(f"Saving difference results to CSV: {val_diff_result_path}/{time_stamp}_diff.csv")
