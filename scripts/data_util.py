import pandas as pd
import json
import zipfile
from glob import glob
import os
import gzip
from tqdm import tqdm
import numpy as np
from datetime import timedelta

def map_caseRate_ZCTA_to_CT():
    """
    DEPRECATED
    :return: no return, write a csv under data
    """
    caseRate = pd.read_csv('../data/caserate-by-modzcta.csv')

    # gather column, rename, parse zip code
    caseRate = pd.melt(caseRate, id_vars = 'week_ending')
    caseRate['variable'] = caseRate['variable'].str.split('_', expand = True).loc[:,1]
    caseRate = caseRate.rename({'variable': 'ZIP', 'value': 'caseRate'}, axis=1)
    caseRate.ZIP = caseRate.ZIP.astype('str')

    # exclude aggregated statistics row
    caseRate = caseRate.loc[~caseRate.ZIP.isin(['SI','QN','MN','BK','BX','CITY']),:]

    crosswalk = pd.read_excel('../data/ZIP_TRACT_crosswalk.xlsx')
    crosswalk['ZIP'] = crosswalk['ZIP'].astype('str').str.pad(5, side='left', fillchar='0')

    # join tables
    caseRate_cw = caseRate.merge(crosswalk, how = 'left', on = 'ZIP')
    caseRate_cw['rateShare'] = caseRate_cw['caseRate'] * caseRate_cw['TOT_RATIO']

    # aggregate to census tract
    caseRateByCT = caseRate_cw.groupby(['week_ending','TRACT']).agg(rate = ('rateShare','sum'))
    caseRateByCT = caseRateByCT.reset_index()

    caseRateByCT.to_csv('../data/caserate_by_tract.csv', index = False)


def write_caseRate_by_ZCTA():
    """

    :return: no return, write a csv under data
    """
    caseRate = pd.read_csv('../data/caserate-by-modzcta.csv')

    # gather column, rename, parse zip code
    caseRate = pd.melt(caseRate, id_vars = 'week_ending')
    caseRate['variable'] = caseRate['variable'].str.split('_', expand = True).loc[:,1]
    caseRate = caseRate.rename({'variable': 'ZIP', 'value': 'caseRate'}, axis=1)
    caseRate.ZIP = caseRate.ZIP.astype('str')

    # exclude aggregated statistics row
    caseRate = caseRate.loc[~caseRate.ZIP.isin(['SI','QN','MN','BK','BX','CITY']),:]
    caseRate['week_ending'] = pd.to_datetime(caseRate['week_ending']) + timedelta(days = 2)
    caseRate['week_ending'] =  caseRate['week_ending'].astype('str')

    caseRate.to_csv('../data/caserate_by_zcta_cleaned.csv', index=False)


def get_NYC_ZIP_List():
    """

    :return: a numpy array containing all ZIP CODES in NYC
    """
    return pd.read_csv('../data/caserate_by_zcta_cleaned.csv')['ZIP'].astype('str').unique()


def write_NYC_crosswalk():

    crosswalk = pd.read_excel('../data/ZIP_TRACT_crosswalk.xlsx')
    crosswalk.ZIP = crosswalk.ZIP.astype('str')
    crosswalk.TRACT = crosswalk.TRACT.astype('str')
    crosswalk = crosswalk.loc[crosswalk.ZIP.isin(get_NYC_ZIP_List()), :]

    crosswalk.to_csv('../data/NYC_ZIP_to_TRACT.csv', index=False)



def unzip_patterns():
    """

    :return: no return, unzip patterns zip to data. (do not push to github folder)
    """
    for file in glob("..\\data\\NY-PATTERNS*.zip"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall('..\\data\\patterns\\{0}'.format(file.split('\\')[2].split('-')[2]))




def clean_patterns_unaggregated(trip_threshold):
    """
    trip_threshold: total count of trips per month. Pairs must have trips larger than trip_threshold to be included.

    :return: no return, write a csv file 'pattern_cleaned.csv' for each month
    """
    cw = pd.read_csv('../data/NYC_ZIP_to_TRACT.csv')
    cw.ZIP, cw.TRACT = cw.ZIP.astype('str'), cw.TRACT.astype('str')

    column_name = ['placekey', 'visits_by_day','visitor_home_aggregation']
    repo = os.listdir('..\\data\\patterns')
    for r in repo:
        file = '../data/patterns/{0}/patterns.csv.gz'.format(r)
        print('Processing {0}'.format(file))
        with gzip.open(file, 'rb') as f_in:
            df = pd.read_csv(f_in)

            # exclude empty dict
            df = df.loc[df['visitor_home_aggregation'] != '{}',column_name]

            # trip count threshold: exclude pairs with less than n visits per month
            # trip_threshold = 30
            l = [json.loads(x) for x in df['visitor_home_aggregation']]
            s = np.array([sum([x[n] for n in x]) for x in l])
            df = df.iloc[np.argwhere(s > trip_threshold).flatten(),:].reset_index(drop = True)

        # loop through rows to map ZIP to TRACT
        for i in tqdm(range(df.shape[0])):
            dict = pd.DataFrame.from_dict(json.loads(df.loc[i, 'visitor_home_aggregation']), orient = 'index')\
                .reset_index()\
                .rename(columns = {'index':'TRACT', 0:'n'})\
                .merge(cw, how='left', on='TRACT').loc[:, ['ZIP', 'n']]\
                .dropna()\
                .groupby('ZIP').agg({'n':'sum'}).reset_index()

            dict = dict.loc[dict.n > 4, :]
            print(dict.n.sum())
            if dict.n.sum() > trip_threshold:

                df['visitor_home_aggregation'][i] = dict.to_dict(orient = 'records')
            else:
                df['visitor_home_aggregation'][i] = ''


        df = df.loc[df['visitor_home_aggregation'] != '',column_name]
        df.to_csv('..\\data\\patterns\\patterns_{0}.csv'.format(r), index=False)
        print('Process {0} Success'.format(file))



def clean_patterns_zip():
    pass


