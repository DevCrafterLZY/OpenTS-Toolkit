# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from tools import gz2csv

if __name__ == "__main__":
    '''
    解压demo
    
    target_columns_map中的每个键对应一个目标CSV文件的名称，
    每个键的值是一个列名的列表，表示该目标文件应包含的指标列。

    tips: 只要文件中包含了该列表的子集，就会被保存到对应的CSV文件中。
    '''
    target_columns_map = {
        "all_detail_detect_label_metrics.csv": ["accuracy", "f_score", "precision", "recall", "adjust_accuracy",
                                         "adjust_f_score", "adjust_precision", "adjust_recall", "rrecall",
                                         "rprecision", "precision_at_k", "rf", "affiliation_f", "affiliation_precision",
                                         "affiliation_recall", 'typical_anomaly_ratio'],
        "all_detail_detect_score_metrics.csv": ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR",
                                         'typical_anomaly_ratio'],
        "all_detail_forcast_metrics.csv": ["mae", "mse", "rmse", "mape", "smape", "mase", "wape", "msmape", "mae_norm",
                                    "mse_norm", "rmse_norm", "mape_norm", "smape_norm", "mase_norm", "wape_norm",
                                    "msmape_norm"],
    }
    log_data = gz2csv(r'demo_result', target_columns_map)
    for file_name, data in log_data.items():
        data.to_csv(file_name, index=False)

    '''
    寻找最好的结果demo

    target_columns_map中的每个键对应一个目标CSV文件的名称，
    每个键的值是一个列名的列表，表示该目标文件应包含的指标列。

    tips: 只要文件中包含了该列表的子集，就会被保存到对应的CSV文件中。
    '''
