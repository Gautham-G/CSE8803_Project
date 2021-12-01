import pandas as pd
import json


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

    caseRate.to_csv('../data/caserate_by_zcta_cleaned.csv', index=False)
    
