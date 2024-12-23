import json
import os

import pandas as pd

MODEL_NAME_MAP = {
    "CATCH": "self_impl.CATCH",
    "ModernTCN": "self_impl.ModernTCN",
    "iTransformer": "time_series_library.iTransformer",
    "DualTF": "self_impl.DualTF",
    "AnomalyTransformer": "self_impl.AnomalyTransformer",
    "DCdetector": "self_impl.DCdetector",
    "TimesNet": "time_series_library.TimesNet",
    "PatchTST": "time_series_library.PatchTST",
    "DLinear": "time_series_library.DLinear",
    "NLinear": "time_series_library.NLinear",

    "AutoEncoder": "merlion.AutoEncoder",
    "ocsvmski": "tods.ocsvmski",
    "isolationforestski": "tods.isolationforestski",
    "IsolationForest": "merlion.IsolationForest",
    "pcaodetectorski": "tods.pcaodetectorski",
    "hbosski": "tods.hbosski",
}

MODEL_ADAPTER = {
    "iTransformer": "transformer_adapter",
    "TimesNet": "transformer_adapter",
    "PatchTST": "transformer_adapter",
    "DLinear": "transformer_adapter",
    "NLinear": "transformer_adapter",
}


def process_model_params(model_params, row):
    model_params = json.loads(model_params)
    if model_params.get("anomaly_ratio", None) is not None:
        return json.dumps(model_params)
    if row["typical_anomaly_ratio"] == "None":
        return json.dumps(model_params)
    model_params["anomaly_ratio"] = float(row["typical_anomaly_ratio"])
    return json.dumps(model_params)


def generate_label_script(row):
    model_name = row['model_name']
    model_module_name = MODEL_NAME_MAP[model_name]
    model_params = row['model_params']
    model_params = process_model_params(model_params, row)
    file_name = row["file_name"]
    adapter = MODEL_ADAPTER.get(model_name, None)
    adapter_part = f' --adapter "{adapter}"' if adapter is not None else ""

    script = (
        f'python ./scripts/run_benchmark.py '
        f'--config-path "unfixed_detect_label_multi_config.json" '
        f'--data-name-list "{file_name}" '
        f'--model-name "{model_module_name}" '
        f'--model-hyper-params \'{model_params}\'{adapter_part} '
        f'--gpus 0 --num-workers 1 --timeout 60000 '
        f'--save-path "label/{model_name}"'
    )

    return script


def generate_score_script(row):
    model_name = row['model_name']
    model_module_name = MODEL_NAME_MAP[model_name]
    model_params = row['model_params']
    file_name = row["file_name"]
    adapter = MODEL_ADAPTER.get(model_name, None)
    adapter_part = f' --adapter "{adapter}"' if adapter is not None else ""
    script = (
        f'python ./scripts/run_benchmark.py '
        f'--config-path "unfixed_detect_score_multi_config.json" '
        f'--data-name-list "{file_name}" '
        f'--model-name "{model_module_name}" '
        f'--model-hyper-params \'{model_params}\'{adapter_part} '
        f'--gpus 0 --num-workers 1 --timeout 60000 '
        f'--save-path "score/{model_name}"'
    )

    return script


if __name__ == '__main__':
    score_result = pd.read_csv('detailed_score_result.csv')
    label_result = pd.read_csv('detailed_label_result.csv')

    for index, row in score_result.iterrows():
        score_script = generate_score_script(row)
        model_name = row["model_name"]
        file_name = row["file_name"]
        dataset_name = os.path.splitext(file_name)[0]
        path = f'multivariate_detection/detect_score/{dataset_name}_script'
        os.makedirs(path, exist_ok=True)
        script_filename = f'{path}/{model_name}.sh'
        with open(script_filename, 'w', newline='\n') as score_file:
            score_file.write(score_script)
            score_file.write('\n')

    for index, row in label_result.iterrows():
        label_script = generate_label_script(row)
        model_name = row["model_name"]
        file_name = row["file_name"]
        dataset_name = os.path.splitext(file_name)[0]
        path = f'multivariate_detection/detect_label/{dataset_name}_script'
        os.makedirs(path, exist_ok=True)
        script_filename = f'{path}/{model_name}.sh'
        with open(script_filename, 'w', newline='\n') as label_file:
            label_file.write(label_script)
            label_file.write('\n')
