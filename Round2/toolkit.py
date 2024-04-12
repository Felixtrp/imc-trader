import pandas as pd
import numpy as np

def filepath_to_csv(filepath):
    csv_df = pd.read_csv(filepath)
    column_names = csv_df.columns[0].split(";")
    title_col = csv_df.columns[0]
    
    data_lists = []
    for column in column_names:
        data_lists.append([])

    for i in range(len(csv_df)):
        data_row = csv_df.iloc[i][title_col].split(";")
        for j in range(len(data_row)):
            data_lists[j].append(data_row[j])

    csv_dict = {}
    for j in range(len(data_row)):
        csv_dict[column_names[j]] = data_lists[j]

    df = pd.DataFrame(csv_dict)

    df['day'] = df['day'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    df['product'] = df['product'].astype(str)

    for i in range(3):
        ask_price_str = "ask_price_" + str(i+1)
        ask_volume_str = "ask_volume_" + str(i+1)
        bid_price_str = "bid_price_" + str(i+1)
        bid_volume_str = "bid_volume_" + str(i+1)

        df[ask_price_str] = df[ask_price_str].replace("", np.nan).astype(float)
        df[ask_volume_str] = df[ask_volume_str].replace("", 0).astype(int)
        df[bid_price_str] = df[bid_price_str].replace("", np.nan).astype(float)
        df[bid_volume_str] = df[bid_volume_str].replace("", 0).astype(int)

    df["mid_price"] = df["mid_price"].replace("", np.nan).astype(float)
    df["profit_and_loss"] = df["profit_and_loss"].replace("", np.nan).astype(float)

    return df

def Orchids_filepath_to_csv(filepath):
    csv_df = pd.read_csv(filepath)
    column_names = csv_df.columns[0].split(";")
    title_col = csv_df.columns[0]
    
    data_lists = []
    for column in column_names:
        data_lists.append([])

    for i in range(len(csv_df)):
        data_row = csv_df.iloc[i][title_col].split(";")
        for j in range(len(data_row)):
            data_lists[j].append(data_row[j])

    csv_dict = {}
    for j in range(len(data_row)):
        csv_dict[column_names[j]] = data_lists[j]

    df = pd.DataFrame(csv_dict)

def Orchids_filepath_to_csv(filepath):
    csv_df = pd.read_csv(filepath)
    column_names = csv_df.columns[0].split(";")
    title_col = csv_df.columns[0]
    
    data_lists = []
    for column in column_names:
        data_lists.append([])

    for i in range(len(csv_df)):
        data_row = csv_df.iloc[i][title_col].split(";")
        for j in range(len(data_row)):
            data_lists[j].append(data_row[j])

    csv_dict = {}
    for j in range(len(data_row)):
        csv_dict[column_names[j]] = data_lists[j]

    df = pd.DataFrame(csv_dict)

    df['DAY'] = df['DAY'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    df['ORCHIDS'] = df['ORCHIDS'].astype(float)
    for i in list(df.columns):
        if i != 'DAY' and i != 'timestamp':
            df[i] = df[i].astype(float)

    return df