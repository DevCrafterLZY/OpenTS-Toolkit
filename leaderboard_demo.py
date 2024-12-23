from tools import csv2best
from tools import data2leaderboard
from tools import leaderboard2data
from tools import result2sh

if __name__ == '__main__':
    file_name = 'result/all_detail_forcast_metrics.csv'
    best_res_name, detail_res_name = csv2best(file_name, 'mae_norm')
    leaderboard = data2leaderboard(detail_res_name)

    detail_selected_res_name = leaderboard2data(file_name, leaderboard)
    result2sh(detail_selected_res_name)