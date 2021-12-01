import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import optparse
from datetime import datetime

start_week = '2020-08-10'
curr_week = '2021-09-27'
ahead = 1

def week_to_number(week: str):
    start_date = datetime.strptime(start_week, '%Y-%m-%d')
    curr_date = datetime.strptime(week, '%Y-%m-%d')
    return (curr_date - start_date).days // 7

curr_week_num = week_to_number(curr_week)

# Get zip time-series data
df = pd.read_csv('data/caserate_by_zcta_cleaned.csv')
df["week_no"] = df["week_ending"].apply(lambda x: week_to_number(x))

# Make date dict
date_dict = {}
ct = 0
for i in df["ZIP"].unique():
    date_dict[i] = ct
    ct += 1

# Make timeseries arrays
tseries = np.zeros((len(df["ZIP"].unique()), curr_week_num + ahead))
for i in df["ZIP"].unique():
    for j in range(curr_week_num + ahead):
        tseries[date_dict[i], j] = df["caseRate"][(df["ZIP"] == i) & (df["week_no"] == j)].values[0]


# Get visit counts


# Get train data
train_data = tseries[:, :curr_week_num+1]
