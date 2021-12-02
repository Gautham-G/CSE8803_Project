import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import optparse
from datetime import datetime
from glob import glob
import json, ast
from tqdm import tqdm
import pickle
from model.encoders import FFN

start_week = "2020-08-17"
curr_week = "2021-09-27"
ahead = 1


def week_to_number(week: str):
    start_date = datetime.strptime(start_week, "%Y-%m-%d")
    curr_date = datetime.strptime(week, "%Y-%m-%d")
    return (curr_date - start_date).days // 7


curr_week_num = week_to_number(curr_week)

# Get zip time-series data
df = pd.read_csv("data/caserate_by_zcta_cleaned.csv")
df["week_no"] = df["week_ending"].apply(lambda x: week_to_number(x))

# Make date dict
date_dict = {}
ct = 0
for i in df["ZIP"].unique():
    date_dict[i] = ct
    ct += 1


os.makedirs("./saves", exist_ok=True)
with open("./saves/tseries.pkl", "rb") as f:
    tseries = pickle.load(f)

with open("./saves/visit_counts_df.pkl", "rb") as f:
    df1 = pickle.load(f)

with open("./saves/visits_count.pkl", "rb") as f:
    visits_count = pickle.load(f)

intersect_pois = set(visits_count[0].keys())
for i in range(1, len(visits_count)):
    intersect_pois = intersect_pois.intersection(set(visits_count[i].keys()))

poi_dict = {}
ct = 0
for i in intersect_pois:
    poi_dict[i] = ct
    ct += 1


with open("./saves/visit_matrix.pkl", "rb") as f:
    visit_matrix = pickle.load(f)

visit_matrix = th.from_numpy(visit_matrix).float()


class ZipEncoder(nn.Module):
    def __init__(self, embed_sz, num_zip, final_sz, rnn_dim):
        super(ZipEncoder, self).__init__()
        self.embed_layer = nn.Embedding(num_zip, embed_sz)
        self.fc = FFN(embed_sz, [60, 60], final_sz)
        self.rnn = nn.GRU(1, rnn_dim, batch_first=True)
        self.fc2 = FFN(rnn_dim + final_sz, [60, 60], 1)

    def forward(self, zip_no, sequences):
        embed = self.embed_layer(zip_no)
        embed = embed.unsqueeze(1)
        embed = embed.repeat(1, sequences.shape[1], 1)
        x = th.cat([embed, sequences], dim=2)
        x = self.fc(x)
        x, _ = self.rnn(x)
        x = self.fc2(x)
        return x, embed


class PoiEncoder(nn.Module):
    def __init__(self, embed_sz, num_poi) -> None:
        super(PoiEncoder, self).__init__()
        self.embed_layer = nn.Embedding(num_poi, embed_sz)
        self.fc = FFN(embed_sz, [60, 60], 1)

    def forward(self, poi_no):
        embed = self.embed_layer(poi_no)
        embed = embed.unsqueeze(1)
        x = self.fc(embed)
        return x, embed


class WeightsEncoder(nn.Module):
    def __init__(self, embed_sz, final_sz):
        super(WeightsEncoder, self).__init__()
        self.fc_zip = nn.Linear(embed_sz, final_sz)
        self.fc_poi = nn.Linear(embed_sz, final_sz)
        self.lamda = nn.Parameter(th.Tensor(0.1), requires_grad=True)

    def forward(self, poi_embeds, zip_embeds, visit_matrix):
        zip_embeds = self.fc_zip(zip_embeds)
        poi_embeds = self.fc_poi(poi_embeds)
        product_embeds = zip_embeds @ th.transpose(poi_embeds, 0, 1)
        weights = th.exp(self.lamda) * th.sigmoid(product_embeds) * visit_matrix
        return weights

