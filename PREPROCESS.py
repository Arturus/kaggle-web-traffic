# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:55:54 2018

@author: gk
"""

#Do some basic preprocessing to get my data in same format as Kaggle code


#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import os
import pandas as pd
#import numpy as np

#from statsmodels.tsa.seasonal import seasonal_decompose
#stl = seasonal_decompose(x)






def load_my_data(myDataDir):
    """
    Load my data
    """
    files = os.listdir(myDataDir)
    files = [i for i in files if i.endswith('.csv')]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    #Exclude certain cities
    #ignore_list = [] #id's of the cities to ignore
    #files = [i for i in files if i.split(".")[0] not in ignore_list]
    dflist = []
    for ii, ff in enumerate(files):
        df = pd.read_csv(os.path.join(myDataDir,ff))
        dflist += [df]
    df = pd.concat(dflist,sort=False)
    df = df[['id','date','y']]
    df['id'] = df['id'].astype(int)
    return df


def remove_cities(df,remove_id_list):
    """
    Remove blacklisted id's [since some downloaded id's no longer relevant,
    or suspected to not be useful, or be corrupted]
    
    Or just ignore these files when loading data and don't need this
    """
    return df.loc[~df['id'].isin(remove_id_list)]
    

def get_earliest_latest_dates(df):
    """
    Get first and last dates seen across any time series
    """
    earliest = min(df['date'])
    latest = max(df['date'])
    print('earliest date',earliest)
    print('latest date',latest)
    return earliest, latest
    



#def __keep_btwn_dates(df,start_date,end_date):
#    """
#    Excerpt only the data between [inclusive] start and end date.
#    Both dates are formatted as 'YYYY-mm-DD'
#    """
#    len1 = len(df)
#    df = df.loc[(df['date']>=start_date) & (df['date']<=end_date)]
#    df.reset_index(inplace=True,drop=True)
#    len2 = len(df)
#    rows_removed = len1 - len2
#    print('rows_removed:',rows_removed,'of',len1)
#    return df   



def remove_seasonal_blocks(df):
    """
    For places in the data where there are missing gaps of length > 1 seasonality,
    remove 
    """
    return







def do_imputation(df,imputation_method):
    """
    For places in the data where missing gaps are smalle (<7 days),
    just fill in those few missing days with a basic 
    remove 
    """
    
    
    def imputation_small_gaps(df,imputation_method):
        """
        Do missing data imputation using the given forecasting method
        Only use this for short missing segments; do not use for longer ones.
        """
        if imputation_method == 'STL':
            #stl = seasonal_decompose(x)
            df_filled = df
            pass
        else:
            raise Exception('That method not implemented yet')
        return df_filled    
    
    
    def imputation_big_gaps(df):
        """
        Do missing data imputation / removal
        For big gaps [gaps bigger than 1 seasonality]
        """
        df_filled = df
        return df_filled    
    
    
    def imputation__simple(df):
        """
        Juat as placeholder for now,
        fill all missing with zeros,
        or mean or median imputation
        """
        df_filled = df
        return df_filled     
    
    
    
    #First deal with small gaps (missing gaps fewer than e.g. 7 days):
    df = imputation_small_gaps(df,imputation_method)
    
    #Deal with longer gaps [e.g. by removing enough blocks of length S, where
    #S is the seasonality, to completely get rid of gaps]
    #...
    #df = imputation_big_gaps(df)
    
    #Trim start and end of each series/ to align to get in phase
    #df = 
    #...
    
    return df







def format_like_Kaggle(df, myDataDir, start_date=None, end_date=None):
    """
    Take my data and format it exactly as needed to use for the Kaggle seq2seq
    model [requires making train_1.csv, train_2.csv, key_1.csv, key_2.csv]
    [??? or does the seq2seq cTUlly OPEN THE .ZIPS DIRECTLY????????]
    """
    
    
    def make_train_csv(df, save_path, start_date, end_date):
        """
        Make the train_1.csv
        """
        #Rename columns to be as in Kaggle data:
        df.rename(columns={'id':'Page'},inplace=True)
        
        #Get earliest and latest date across all series to align times [pad start/end]
        earliest, latest = get_earliest_latest_dates(df)
        
        #Excerpt only the relevant time interval, if manually specified
        if start_date:
            earliest = max(earliest,start_date)
        if end_date:
            latest = min(latest,end_date)
        
        idx = pd.date_range(earliest,latest)
        OUT_OF_RANGE_FILL_VALUE = -1 #np.NaN #0 #puttign as nan casts to float and cannot convert to int

        #Reorganize data for each id (->"Page")
        unique_ids = pd.unique(df['Page'])
        df_list = []
        for i, u in enumerate(unique_ids):
            d = df.loc[df['Page']==u]
            #Nan / zero pad start and end date range if needed {end missing}
            dates = pd.Series(d['y'].values,index=d['date'])
            dates.index = pd.DatetimeIndex(dates.index)        
            dates = dates.reindex(idx, fill_value=OUT_OF_RANGE_FILL_VALUE)
            dates.index = pd.to_datetime(dates.index).strftime('%Y-%m-%d')
            dd = pd.DataFrame(dates).T 
            dd['Page'] = u
            df_list.append(dd)
        
        df = pd.concat(df_list,axis=0)
        cols = df.columns.tolist()
        df = df[cols[-1:]+cols[:-1]]
        df.reset_index(drop=True,inplace=True)
        df.to_csv(save_path,index=False)
        return df
    
    
    def make_key_csv(df):
        """
        Make the key_1.csv, key_2.csv
        May actually not need this???
        """
        #save out
        return    
    
    
    #Make the train csv [for now just do 1, ignore the train 2 part ???]
    save_path = os.path.join(os.path.split(myDataDir)[0],'train_1_my_data.csv')
    df = make_train_csv(df, save_path, start_date, end_date)

    #For the prediction phase, need the key ????
#    make_key_csv(df)
    
    return











if __name__ == '__main__':
    
    # =============================================================================
    #     PARAMETERS
    # =============================================================================
    # TOTAL COMPLETED TRIPS:
    myDataDir = r"/Users/......../Desktop/exData/totalCompletedTripsDaily"
    imputation_method = 'STL'
    START_DATE = '2015-01-01' #None
    END_DATE = '2017-12-31' #None
    REMOVE_ID_LIST = []#[3,4]#id's for locations that are no longer useful



    # =============================================================================
    #     MAIN
    # =============================================================================
    print('START_DATE',START_DATE)
    print('END_DATE',END_DATE)
    print('REMOVE_ID_LIST',REMOVE_ID_LIST)
    print('imputation_method',imputation_method)
    print('myDataDir',myDataDir)
    
    #Load
    df = load_my_data(myDataDir)
    
    #Remove any bad/irrelevant cities
    df = remove_cities(df,REMOVE_ID_LIST)
    
    #Put into same format as used by Kaggle, save out csv's    
    format_like_Kaggle(df, myDataDir, start_date=START_DATE, end_date=END_DATE)
    
    
    #Imputation, dealing with missing seasonality blocks / out of phase
    df = do_imputation(df,imputation_method)