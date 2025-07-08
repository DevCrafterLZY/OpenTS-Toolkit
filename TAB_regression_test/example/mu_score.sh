PYTHON_EXEC=$1
ROOT_PATH=$2
SAVE_PATH=$3

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.AnomalyTransformer"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "lr": 0.001, "num_epochs": 3, "win_size": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/AnomalyTransformer"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.AutoEncoder"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/AutoEncoder"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.CATCH"  --data-name-list "NYC.csv" --model-hyper-params '{"Mlr": 1e-05, "auxi_lambda": 0.1, "batch_size": 64, "cf_dim": 32, "d_ff": 64, "d_model": 16, "dc_lambda": 0.1, "e_layers": 3, "head_dim": 32, "lr": 0.0001, "n_heads": 16, "num_epochs": 3, "patch_size": 16, "patch_stride": 8, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/CATCH"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.cblofski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/cblofski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.ContraAD"  --data-name-list "NYC.csv" --model-hyper-params '{"n_window": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/ContraAD"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.DAGMM"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DAGMM"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.DCdetector"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 128, "num_epochs": 3, "patch_size": [3, 5, 7], "win_size": 105}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DCdetector"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.DeepPointAnomalyDetector"  --data-name-list "NYC.csv" --model-hyper-params '{"enable_threshold": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DeepPointAnomalyDetector"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.DLinear"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 10, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DLinear"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.DualTF"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 8}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DualTF"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "duet.DUET"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 256, "horizon": 1, "lr": 0.0001, "norm": true, "num_epochs": 10, "seq_len": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DUET"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.EIF"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/EIF"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.hbosski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/hbosski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.IsolationForest"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/IsolationForest"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.iTransformer"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 128, "e_layers": 2, "horizon": 0, "norm": true, "num_epochs": 5, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/iTransformer"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.KMeans"  --data-name-list "NYC.csv" --model-hyper-params '{"window_size": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/KMeans"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.knnski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/knnski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.lodaski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/lodaski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.lofski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/lofski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.LSTMED"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/LSTMED"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.ModernTCN"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "dims": [32], "dropout": 0.2, "ffn_ratio": 1, "head_dropout": 0.0, "horizon": 100, "itr": 1, "label_len": 0, "large_size": [71], "lr": 0.001, "num_blocks": [1], "num_epochs": 1, "patch_size": 8, "patch_stride": 4, "patience": 10, "seq_len": 100, "small_kernel_merged": false, "small_size": [5], "use_multi_scale": false}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/ModernTCN"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.NLinear"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "d_ff": 128, "d_model": 128, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 10, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/NLinear"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.ocsvmski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/ocsvmski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.PatchTST"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 32, "d_ff": 128, "d_model": 128, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 10, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/PatchTST"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "tods.pcaodetectorski"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/pcaodetectorski"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "time_series_library.TimesNet"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 32, "d_ff": 64, "d_model": 64, "e_layers": 2, "horizon": 0, "norm": true, "num_epochs": 10, "seq_len": 100}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimesNet"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.Torsk"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Torsk"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "self_impl.TranAD"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 64, "lr": 0.0001, "n_window": 100, "num_epochs": 5, "patience": 3}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TranAD"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "merlion.VAE"  --data-name-list "NYC.csv" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/VAE"


"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.CALFModel"  --data-name-list "NYC.csv" --model-hyper-params '{"d_ff": 768, "d_model": 768, "dataset": "synthetic", "dropout": 0.3, "gpt_layer": 6, "horizon": 1, "lr": 0.0005, "n_heads": 4, "norm": true, "sampling_rate": 0.05, "seq_len": 96}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/CALFModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.CALFModel"  --data-name-list "NYC.csv" --model-hyper-params '{"d_ff": 768, "d_model": 768, "dropout": 0.3, "gpt_layer": 6, "horizon": 1, "lr": 0.0005, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/CALFModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Chronos"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 0.05, "seq_len": 96}' --adapter "chronos_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Chronosfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Chronos"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 96}' --adapter "chronos_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Chronosfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Chronos"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 0, "norm": true, "seq_len": 96}' --adapter "chronos_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Chronoszero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.DadaModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "lr": 0.005, "norm": true, "sampling_rate": 0.05, "seq_len": 100}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DadaModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.DadaModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 100}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DadaModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.DadaModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "norm": true, "seq_len": 100}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/DadaModelzero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.GPT4TSModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "norm": true, "num_epochs": 3, "sampling_rate": 0.05, "seq_len": 100}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/GPT4TSModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.GPT4TSModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 100}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/GPT4TSModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.LLMMixerModel"  --data-name-list "NYC.csv" --model-hyper-params '{"d_model": 32, "horizon": 1, "lr": 0.001, "n_heads": 4, "norm": true, "sampling_rate": 0.05, "seq_len": 96, "use_norm": 1}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/LLMMixerModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.LLMMixerModel"  --data-name-list "NYC.csv" --model-hyper-params '{"d_model": 32, "horizon": 1, "lr": 0.001, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96, "use_norm": 1}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/LLMMixerModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Moment"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 0.05, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Momentfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Moment"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "num_epochs": 3, "sampling_rate": 1, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Momentfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.Moment"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 0, "norm": true, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/Momentzero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimerModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 0.05, "seq_len": 672}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimerModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimerModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 672}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimerModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimerModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 0, "norm": true, "seq_len": 672}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimerModelzero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimesFM"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "input_patch_len": 32, "is_train": 1, "norm": true, "output_patch_len": 128, "sampling_rate": 0.05, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimesFMfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimesFM"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "input_patch_len": 32, "is_train": 1, "norm": true, "output_patch_len": 128, "sampling_rate": 1, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimesFMfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TimesFM"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "input_patch_len": 32, "is_train": 0, "norm": true, "output_patch_len": 128, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TimesFMzero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TinyTimeMixer"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TinyTimeMixerfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TinyTimeMixer"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TinyTimeMixerfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.TinyTimeMixer"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "horizonon": 1, "norm": true, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/TinyTimeMixerzero"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.UniTimeModel"  --data-name-list "NYC.csv" --model-hyper-params '{"batch_size": 32, "dataset": "UV", "horizon": 1, "max_backcast_len": 96, "max_token_num": 17, "norm": true, "sampling_rate": 0.05, "seq_len": 96}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/UniTimeModelfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.UniTimeModel"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "dataset": "UV", "max_backcast_len": 512, "max_token_num": 80, "norm": true, "sampling_rate": 1, "seq_len": 512, "stride": 16}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/UniTimeModelfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.UniTS"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 0.05, "seq_len": 96, "target_dim": 19}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/UniTSfew"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.UniTS"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 96, "target_dim": 19}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/UniTSfull"

"$PYTHON_EXEC" "$ROOT_PATH"/scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "pre_train.UniTS"  --data-name-list "NYC.csv" --model-hyper-params '{"horizon": 1, "is_train": 0, "norm": true, "seq_len": 96, "target_dim": 19}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "$SAVE_PATH/UniTSzero"
