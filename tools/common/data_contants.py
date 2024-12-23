DATA_SET_NAMES = [
    "ASD_dataset_1.csv", "ASD_dataset_2.csv", "ASD_dataset_3.csv", "ASD_dataset_4.csv",
    "ASD_dataset_5.csv", "ASD_dataset_6.csv", "ASD_dataset_7.csv", "ASD_dataset_8.csv",
    "ASD_dataset_9.csv", "ASD_dataset_10.csv", "ASD_dataset_11.csv", "ASD_dataset_12.csv",

    "daphnet_S01R02.csv", "daphnet_S02R01.csv", "daphnet_S03R01.csv", "daphnet_S03R02.csv",
    "daphnet_S07R01.csv", "daphnet_S07R02.csv", "daphnet_S08R01.csv",

    "Exathlon_4_1_100000_61-32.csv", "Exathlon_5_1_100000_63-64.csv", "Exathlon_5_1_100000_64-63.csv",

    "SKAB_other_10.csv", "SKAB_other_11.csv", "SKAB_other_12.csv", "SKAB_other_13.csv",
    "SKAB_other_14.csv", "SKAB_other_2.csv", "SKAB_other_3.csv", "SKAB_other_4.csv",
    "SKAB_other_5.csv", "SKAB_other_6.csv", "SKAB_other_7.csv", "SKAB_other_8.csv",
    "SKAB_other_9.csv", "SKAB_valve1_0.csv", "SKAB_valve1_1.csv", "SKAB_valve1_10.csv",
    "SKAB_valve1_11.csv", "SKAB_valve1_12.csv", "SKAB_valve1_13.csv", "SKAB_valve1_14.csv",
    "SKAB_valve1_15.csv", "SKAB_valve1_2.csv", "SKAB_valve1_3.csv", "SKAB_valve1_4.csv",
    "SKAB_valve1_5.csv", "SKAB_valve1_6.csv", "SKAB_valve1_7.csv", "SKAB_valve1_8.csv",
    "SKAB_valve1_9.csv", "SKAB_valve2_0.csv", "SKAB_valve2_1.csv", "SKAB_valve2_2.csv",
    "SKAB_valve2_3.csv", "SKAB_other_1.csv",

    "SMAP.csv", "SWAN.csv", "SWAT.csv", "PUMP.csv",
    "CalIt2.csv", "CICIDS.csv", "Creditcard.csv", "GECCO.csv",
    "Genesis.csv", "MSL.csv", "NYC.csv", "PSM.csv",
    "SMD.csv",
]

SYNC_DATA_SET_NAMES = [
    "synthetic_con0.072.csv", "synthetic_con0.0494.csv", "synthetic_glo0.048.csv", "synthetic_glo0.0718.csv",
    "synthetic_sea0.0482.csv", "synthetic_sea0.0774.csv", "synthetic_sha0.049.csv", "synthetic_sha0.0742.csv",
    "synthetic_sub_mix0.089.csv", "synthetic_sub_mix0.0574.csv", "synthetic_tre0.0482.csv", "synthetic_tre0.0778.csv",
]

M_DATASET_FILE_NAMES = DATA_SET_NAMES + SYNC_DATA_SET_NAMES

Model_NAMES = [
    "self_impl.AnomalyTransformer", "self_impl.DCdetector", "self_impl.DualTF",
    "self_impl.MatrixProfile", "self_impl.TranAD",

    "time_series_library.DLinear", "time_series_library.TimesNet", "time_series_library.PatchTST",
    "time_series_library.NLinear", "time_series_library.iTransformer",

    "merlion.AutoEncoder", "merlion.DeepPointAnomalyDetector", "merlion.IsolationForest",
    "merlion.LSTMED", "merlion.VAE", "self_impl.ModernTCN",
    "merlion.ArimaDetector", "merlion.DAGMM", "merlion.SarimaDetector",
    "merlion.SpectralResidual", "merlion.StatThreshold", "merlion.ZMS",

    "tods.hbosski", "tods.knnski", "tods.lodaski",
    "tods.lofski", "tods.ocsvmski", "tods.pcaodetectorski",

    "GPT4TSModel", "Moment", "TinyTimeMixer",
    "UniTimeModel", "UniTS"
]

M_Model_NAMES = [
    "self_impl.AnomalyTransformer", "self_impl.DCdetector", "self_impl.DualTF",
    "self_impl.MatrixProfile", "self_impl.TranAD",

    "time_series_library.DLinear", "time_series_library.TimesNet", "time_series_library.PatchTST",
    "time_series_library.NLinear", "time_series_library.iTransformer",

    "merlion.AutoEncoder", "merlion.DeepPointAnomalyDetector", "merlion.IsolationForest",
    "merlion.LSTMED", "merlixon.VAE", "self_impl.ModernTCN",
    "merlion.DAGMM",

    "tods.hbosski", "tods.knnski", "tods.lodaski",
    "tods.lofski", "tods.ocsvmski", "tods.pcaodetectorski",

    "GPT4TSModel", "Moment", "TinyTimeMixer",
    "UniTimeModel", "UniTS"
]

METRIC_MAP = {
    "accuracy": "Acc",
    "precision": "P",
    "recall": "R",
    "f_score": "F1",
    "rprecision": "R-P",
    "rrecall": "R-R",
    "rf": "R-F",
    "affiliation_precision": "Aff-P",
    "affiliation_recall": "Aff-R",
    "affiliation_f": "Aff-F",
    "auc_roc": "A-R",
    "auc_pr": "PR",
    "R_AUC_ROC": "R-A-R",
    "R_AUC_PR": "R-A-P",
    "VUS_ROC": "V-ROC",
    "VUS_PR": "V-PR",
}

MODEL_MAP = {
    "CATCH": "CATCH",
    "ModernTCN": "Modern",
    "iTransformer": "iTrans",
    "DualTF": "DualTF",
    "AnomalyTransformer": "ATrans",
    "DCdetector": "DC",
    "TimesNet": "TsNet",
    "PatchTST": "Patch",
    "DLinear": "DLin",
    "NLinear": "NLin",
    "AutoEncoder": "AE",
    "ocsvmski": "Ocsvm",
    "IsolationForest": "IF",
    "isolationforestski": "IF",
    "pcaodetectorski": "PCA",
    "hbosski": "HBOS",
}

DATA_NAME2CSV_MAP = {
    "CICIDS": "CICIDS.csv",
    "CalIt2": "CalIt2.csv",
    "Credit": "Creditcard.csv",
    "GECCO": "GECCO.csv",
    "Genesis": "Genesis.csv",
    "MSL": "MSL.csv",
    "NYC": "NYC.csv",
    "PSM": "PSM.csv",
    "SMD": "SMD.csv",

    "contextual4.94": "synthetic_con0.0494.csv",
    "contextual7.2": "synthetic_con0.072.csv",
    "global4.8": "synthetic_glo0.048.csv",
    "global7.18": "synthetic_glo0.0718.csv",
    "season4.82": "synthetic_sea0.0482.csv",
    "season7.72": "synthetic_sea0.0774.csv",
    "shapelet4.9": "synthetic_sha0.049.csv",
    "shapelet7.42": "synthetic_sha0.0742.csv",
    "sub_mix5.74": "synthetic_sub_mix0.0574.csv",
    "sub_mix8.9": "synthetic_sub_mix0.089.csv",
    "trend4.82": "synthetic_tre0.0482.csv",
    "trend7.78": "synthetic_tre0.0778.csv",
}

CSV2DATA_NAME_MAP = {
    "CICIDS.csv": "CICIDS",
    "CalIt2.csv": "CalIt2",
    "Creditcard.csv": "Credit",
    "GECCO.csv": "GECCO",
    "Genesis.csv": "Genesis",
    "MSL.csv": "MSL",
    "NYC.csv": "NYC",
    "PSM.csv": "PSM",
    "SMD.csv": "SMD",

    "synthetic_con0.0494.csv": "Contextual4.9",
    "synthetic_con0.072.csv": "Contextual7.2",
    "synthetic_glo0.048.csv": "Global4.8",
    "synthetic_glo0.0718.csv": "Global7.2",
    "synthetic_sea0.0482.csv": "Seasonal4.8",
    "synthetic_sea0.0774.csv": "Seasonal7.7",
    "synthetic_sha0.049.csv": "Shapelet4.9",
    "synthetic_sha0.0742.csv": "Shapelet7.4",
    "synthetic_sub_mix0.0574.csv": "Mixture5.7",
    "synthetic_sub_mix0.089.csv": "Mixture5.9",
    "synthetic_tre0.0482.csv": "Trend4.8",
    "synthetic_tre0.0778.csv": "Trend7.8",
}
