# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:55:54 2018

@author: gk
"""

#Do some basic preprocessing to get my data in same format as Kaggle code


#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

#from statsmodels.tsa.seasonal import seasonal_decompose
#stl = seasonal_decompose(x)

from sklearn.preprocessing import Imputer
from collections import Counter




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





def __missing_vals_distribution(df):
    """
    Look at two things:
        - What fraction of our time series are desne vs. have >= 1 missing value?
        - Of the series that have missing values, what is distribution of gap lengths?
          [important to know since will be doing imputation on it]
    
    df - in format like Kaggle competition: cols are dates, rows are series
         start/end missing, nd intermedite gaps have been filled with -1
    """

    def make_cdf(v):
        c = Counter(v)
        x = c.keys()
        x = np.array(x) - 1 #-1 to go from diff in days from present data -> gap length
        y = c.values()
    #    print(c)
        plt.figure()
        #plt.plot(x,y,drawstyle='steps')#,marker='o')
        plt.plot(x,y,linestyle='None',marker='o')
        plt.title('Distribution of Missing Data Gap Length',fontsize=20)
        plt.xlabel('Gap Length [days]',fontsize=20)
        plt.ylabel('Count',fontsize=20)
    #    plt.axis([-1,10,0,550])
        plt.show()

    #get fraction dense vs sparse:
    dd = df.values[:,1:]
    sparse = (dd==-1).sum(axis=1)
    Nsparse = float((sparse>0).sum())
    print(Nsparse)
    Ntotal = float(dd.shape[0])
    fraction_dense = (Ntotal - Nsparse) / Ntotal
    print('Nsparse', Nsparse)
    print('Ntotal', Ntotal)
    print('fraction_dense', fraction_dense)
    
    #Look at distribution of INTERMEDIATE gap lengths
    #ignore the leading / lagging unfilled since could just be from the series 
    #not officially starting yet, or it got closed out.
    all_gaps = []
    for row in dd:
        inds = np.where(row!=-1)[0]
        x = np.diff(inds)
        t = list(x[x>1])
        if len(t)>0:
            all_gaps.extend(t)
    make_cdf(all_gaps)






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
    
    
    def imputation__simple(df,imputation_method):
        """
        Juat as placeholder for now,
        fill all missing with zeros,
        or mean or median imputation
        """
        missing_values = [-1]#['NaN', -1]
        imp = Imputer(missing_values=missing_values,
                strategy=imputation_method,
                axis=1)
        vals = imp.fit_transform(df.values)#[:,1:]) #The data is only [:,1:]. 
        #"Some rows only contain missing values: [ 35 251 281]"
        #But get some rows with all missing vals. Since we don't actualyl care about this and never will use this 
        #for now just use the "Page" number as well to avoid this.
        
        
        cols = df.columns
        new_df = pd.DataFrame({cols[i]:vals[:,i] for i in range(vals.shape[1])})
        new_df['Page'] = df['Page']
        #Put "Page" at left
        cols = new_df.columns.tolist()
        new_df = new_df[cols[-1:]+cols[:-1]]
        new_df.reset_index(drop=True,inplace=True)        
        return new_df   
    
    
    if (imputation_method == 'median') or (imputation_method == 'mean'):
        df = imputation__simple(df,imputation_method)
    
    else:
        raise Exception('not implemented other methods yet')
    
    #First deal with small gaps (missing gaps fewer than e.g. 7 days):
    #df = imputation_small_gaps(df,imputation_method)
    
    #Deal with longer gaps [e.g. by removing enough blocks of length S, where
    #S is the seasonality, to completely get rid of gaps]
    #...
    #df = imputation_big_gaps(df)
    
    #Trim start and end of each series/ to align to get in phase
    #df = 
    #...
    
    return df







def format_like_Kaggle(df, myDataDir, imputation_method, sampling_period, start_date=None, end_date=None):
    """
    Take my data and format it exactly as needed to use for the Kaggle seq2seq
    model [requires making train_1.csv, train_2.csv, key_1.csv, key_2.csv]
    [??? or does the seq2seq cTUlly OPEN THE .ZIPS DIRECTLY????????]
    """
    
    
    def make_train_csv(df, save_path, imputation_method, sampling_period, start_date, end_date):
        """
        Make the train_2.csv
        """
        
        def aggregate(df, sampling_period):
            """
            Aggregate the data (average it) to downsample
            to desired sample period, e.g. daily measurements -> weekly or monthly.
            Should smooth out some noise, and help w seasonality.
            """
            return df
        
        
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
            
            #If doing imputation / other
            #for each series individually
            #...
            
            df_list.append(dd)
        
        df = pd.concat(df_list,axis=0)
        cols = df.columns.tolist()
        df = df[cols[-1:]+cols[:-1]]
        df.reset_index(drop=True,inplace=True)
        
        
        #Just for analysis: look at kinds of gaps in series    
        __missing_vals_distribution(df)     
            
        
        #Imputation, dealing with missing seasonality blocks / out of phase
        df = do_imputation(df,imputation_method)
        #Could do impoutation then downsampling, vs. downsampling then imputation ... unclear which is better here in general.
        #for now assume we do ipmutation THEN aggregation:
        df = aggregate(df,sampling_period)


            
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
    save_path = os.path.join(os.path.split(myDataDir)[0],f"train_2[ours_{sampling_period}].csv")
    df = make_train_csv(df, save_path, imputation_method, sampling_period, start_date, end_date)

    #For the prediction phase, need the key ????
#    make_key_csv(df)
    
    
   
    
    return df











if __name__ == '__main__':
    
    # =============================================================================
    #     PARAMETERS
    # =============================================================================
    # TOTAL COMPLETED TRIPS:
    myDataDir = r"/Users/kocher/Desktop/forecasting/exData/totalCompletedTripsDaily"
    imputation_method = 'median' #'STL'
    START_DATE = '2015-01-01' #None
    END_DATE = '2017-12-31' #None
    REMOVE_ID_LIST = []#[3,4]#id's for locations that are no longer useful
    SAMPLING_PERIOD = 'daily' #'daily', 'weekly', 'monthly'



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
    df = format_like_Kaggle(df, myDataDir, imputation_method, SAMPLING_PERIOD, start_date=START_DATE, end_date=END_DATE)
    

