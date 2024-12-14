if __name__ == '__main__':
    file_name = 'all_res_10_29/all_res_10_29.csv'
    best_res_name = get_best_res(file_name)
    res_xlsx = tidy_res(best_res_name)
    detail_selected_res_name = find_record(file_name, res_xlsx, "Sheet1")
    result2sh(detail_selected_res_name)