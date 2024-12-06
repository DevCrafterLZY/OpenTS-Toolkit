import pandas as pd

meta_data = pd.read_csv("DETECT_META.csv")

data_set_file_names = meta_data["file_name"].unique()
print(data_set_file_names)