import pandas as pd
import json


def map_caseRate_ZCTA_to_CT():
    caseRate = pd.read_csv('../data/caserate-by-modzcta.csv')
    crosswalk = pd.read_csv('../data/')

    # gather column, rename, parse zip code
    caseRate = pd.melt(caseRate, id_vars = 'week_ending')
    caseRate['variable'] = caseRate['variable'].str.split('_', expand = True).loc[:,1]
    caseRate = caseRate.rename({'variable': 'zip', 'value': 'caseRate'}, axis=1)

    # exclude aggregated statistics row
    caseRate = caseRate.loc[~caseRate.zip.isin(['SI','QN','MN','BK','BX','CITY']),:]

