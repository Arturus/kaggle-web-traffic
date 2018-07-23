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

from copy import deepcopy
from scipy.signal import medfilt


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





def __missing_vals_distribution(df,out_of_range_fill_value):
    """
    Look at two things:
        - What fraction of our time series are desne vs. have >= 1 missing value?
        - Of the series that have missing values, what is distribution of gap lengths?
          [important to know since will be doing imputation on it]
    
    df - in format like Kaggle competition: cols are dates, rows are series
         start/end missing, nd intermedite gaps have been filled with -1
    """

    def make_cdf(v,out_of_range_fill_value):
        c = Counter(v)
        x = list(c.keys())
        x = np.array(x) -1 #-1 to go from diff in days from present data -> gap length
        y = list(c.values())
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
    sparse = (dd==out_of_range_fill_value).sum(axis=1)
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
        inds = np.where(row!=out_of_range_fill_value)[0]
        x = np.diff(inds)
        t = list(x[x>1])
        if len(t)>0:
            all_gaps.extend(t)
    make_cdf(all_gaps,out_of_range_fill_value)






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
        
#    if imputation_method == 'lagKmedian':
#        #First get rid of the big blocks of mising values [more than 1 seasonality long]
##        df = imputation_big_gaps(df)
#        #Then deal with the short missing holes
#        N_seasons = 4
#        df = imputation_lagKmedian(df,N_seasons)
        
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





def imputation_lagKmedian_single_series(df,seasonality,N_seasons,out_of_range_fill_value):
    """
    Fill in short missing gaps by replacing missing value with:
        median over last K weeks for that day.
        E.g. Monday is missing, so use median count over 4 previous Mondays
        
    Intended for short holes. Remove longer ones in chunks of length seasonality.
    
    For now assuming that big chunk removal is done AFTER this step.
    """
    #If the whole series is empty (all -1/NAN):    
    if np.alltrue(df.drop(columns='Page').values==out_of_range_fill_value):
        return df

    max_block_length = seasonality - 1
    offsets = np.arange(-max_block_length,1,1)
    
    cols = list(df.columns)
    cols = cols[:-1]#only the date cols., not the "Page" col
#    N_timesteps = len(cols)
#    print(cols)
#    print(N_timesteps)
    c = df['Page'].values
    _ = df.drop(columns=['Page'])
#    print(_.values)
    missing_inds = np.where(_<0.)[1]


    if missing_inds.size > 0:
        #Means there are some missing values
        #So scan through the data and fill in bad values,
        #starting after the first real data [ignore all -1's that occur before
        #time series starts for real]
        first_real_ind = np.where(_>=0.)[1][0]
        missing_inds =  missing_inds[missing_inds>first_real_ind]
#        print(missing_inds)
        
        for mi in missing_inds:
            #Only fill in those gaps that are small holes (less than 1 seasonality)
            #Check that this particular missing val is not in a missing block 
            #that has >= 1 seasonality size:
#            print(mi)
            in_block = False
           
            for off in offsets:
#                print(_.values)
                block_inds = np.arange(mi+off,mi+off+seasonality,1)
#                print(block_inds)
#                print(block_inds, [i in missing_inds for i in block_inds])
                if np.alltrue([i in missing_inds for i in block_inds]):
                    in_block = True
                    break
            if in_block:
                continue 
            #If it is not in a completely missing block [at least 1 value is recorded], then do lag K median:
            prev_K_inds = np.arange(mi-seasonality, max(0, mi - N_seasons*(seasonality+1)), -seasonality).tolist()
            t = _[_.columns[prev_K_inds]].values
            t = t[t>=0]
            imputed_val = np.median(t)
            #If all K previous timesteps were -1, then would give nan, so set manually to -1:
            if np.isnan(imputed_val):# == np.nan:
                imputed_val = out_of_range_fill_value
            _[_.columns[mi]] = imputed_val
            
#        g = np.where(_<0.)[1]
#        g = g[g>first_real_ind]
#        print(g)
#        print('\n'*3)
    _['Page'] = c
    return _




def data_augmentation(df, jitter_pcts_list=[.05,.01], do_low_pass_filter=True, additive_trend=False):
    """
    Do some basic data augmentation with a few different options.
    Then output Cartesian product of all these variations as the final set.
    
    Any missing point (NAN) will be left as NAN, but others will be modified in some way
    """
    
    def jitter__uniform_pcts(df, jitter_pcts_list, N_perturbations):
        """
        On each observed value (non-NAN), add +/- jitter up to some
        percent of the observed value. Either positive or negative.
        If the count is small, then just leave it, otherwise perturb
        (always leaving counts positive).
        
        Just do at most 1 or 2 percent jitter to not corrupt to much,
        ~ magnitude of measurement noise.
        """
        page = df['Page'].values[0]
        cols = df.columns
        x = df.drop(columns=['Page']).values[0]
        dflist = []
        for uniform_jitter in jitter_pcts_list:
            ids = [str(page) + '__unijit{}_'.format(str(uniform_jitter)) + str(kk+1) for kk in range(N_perturbations)]
            _ = np.zeros((N_perturbations,x.size))
            f = lambda i: np.random.uniform(-uniform_jitter*i,uniform_jitter*i) if not np.isnan(i) else np.nan
            for kk in range(N_perturbations):
                _[kk] = [i + f(i) for i in x]# ).reshape(1,x.size)
            d = {cols[i]:_[:,i] for i in range(x.size)}
            df = pd.DataFrame(data=d)
            df['Page'] = ids
            dflist += [df]
        df = pd.concat(dflist,axis=0)
        df.reset_index(drop=True,inplace=True)
        return df


    def add_trend(df, slopes_list):
        """
        On each observed value (non-NAN), add +/- X_t, where X_t is from a 
        linear trend with given slope, applied across whole series.
        
        Could change the character of the time series a lot so maybe not so good?
        """
        return df

    def low_pass_filter(df, filter_type, kernel_size):
        """
        Low-pass filter the data with some kind of kernel, with some kernel size.
        
        Is going to smooth out the data a lot, not sure if this will change the
        time series too much to be good??
        """
        page = df['Page'].values[0]
        cols = df.columns
        x = df.drop(columns=['Page']).values[0]
        ids = [str(page) + '__{0}{1}'.format(filter_type.func_name,kernel_size)]
        y = filter_type(x,kernel_size=kernel_size)
        _ = np.where(np.invert(np.isnan(x)),y,np.nan)
        d = {cols[i]:_[i] for i in range(x.size)}
        df = pd.DataFrame(data=d,index=[0])
        df['Page'] = ids
        return df


    #For each method, do 5x random
    N_perturbations = 5
    dflist = [df]
    if jitter_pcts_list:
        dflist += [jitter__uniform_pcts(df, jitter_pcts_list, N_perturbations)]
    if do_low_pass_filter:
        filter_type = medfilt
        kernel_size = 7
        dflist += [low_pass_filter(df, filter_type, kernel_size)]
    if additive_trend:
        slopes_list = [-1.1, 1.1]
        dflist += [add_trend(df, slopes_list)]        
#    if autoencoder:
#        #Run through autoencoder, do VAE, get resulting series
    
    df = pd.concat(dflist,axis=0)    
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
        
        def aggregate_to_weekly(df, aggregation_type):
            """
            Aggregate the data (average it) to downsample
            to desired sample period, e.g. daily measurements -> weekly or monthly.
            Should smooth out some noise, and help w seasonality.
            
            **ASSUMES WE HAVE DAILY DATA TO START.
            """
            dfc = deepcopy(df)
            dfc['month-day'] = dfc['date'].apply(lambda x: str(x)[5:])

            #Differentiate by year
            years = pd.DatetimeIndex(dfc['date']).year
            #years -= years.min()
            dfc['year'] = years
            
            #Manually define as below, as generated by pd.date_range('2015-01-01','2015-12-24',freq='W-THU')
            fixed_start_dates = ['01-01','01-08','01-15','01-22',
            '01-29','02-05','02-12','02-19',
            '02-26','03-05','03-12','03-19',
            '03-26','04-02','04-09','04-16',
            '04-23','04-30','05-07','05-14',
            '05-21','05-28','06-04','06-11',
            '06-18','06-25','07-02','07-09',
            '07-16','07-23','07-30','08-06',
            '08-13','08-20','08-27','09-03',
            '09-10','09-17','09-24','10-01',
            '10-08','10-15','10-22','10-29',
            '11-05','11-12','11-19','11-26',
            '12-03','12-10','12-17','12-24']#This combines last ~10 days of year together


            _ = [np.searchsorted(fixed_start_dates,str(x),side='right') - 1 for x in dfc['month-day'].values]
            _ = np.clip(_,0,51).astype(int) #clip 52 to 51. This means lumping last few days of year into 2nd last week of year starting 12/24.
            _ = [fixed_start_dates[i] for i in _]
            #Overwrite the actual date with the predefined week start date:
            dfc['week_start_date'] = dfc['year'].map(str) + '-' + _
            
            #For each page-year-week, aggregte over the N<=7 days of that week to get the aggregted value:            
#            _ = dfc.groupby(['Page','year','week_start_date']).agg({'y': [aggregation_type,'size'], 'year':'min', 'date':'min', 'Page':'min', 'week_start_date':'min'})
            _ = dfc.groupby(['Page','week_start_date']).agg({'y': [aggregation_type,'size'], 'date':'min', 'Page':'min', 'week_start_date':'min'})
            new_df = pd.DataFrame({'Page': _['Page']['min'].values,
                                   'date': _['date']['min'].values,
                                   'y': _['y'][aggregation_type].values, #This is no longer necessarily an int
                                   'week_start_date': _['week_start_date']['min'].values
                                   })

            #After above process, can still have missing blocks for a given time series, so will deal with them later.

            #now that done, delete uneeded columns
            new_df.drop(columns=['date'],inplace=True)
            new_df.rename(columns={'week_start_date':'date'},inplace=True)
            
            return new_df
        
        
        def remove_downsample_columns(df, out_of_range_fill_value):
            """
            When doing any kind of daily --> weekly or monthly aggregation,
            will have many days that are now empty (all data aggregated to single
            date marking 1st date of week / month)
            
            So remove those obsolete columns
            """
            bad_cols = [i for i in df.columns if np.alltrue(df[i].values==out_of_range_fill_value)]
            df.drop(columns=bad_cols,inplace=True)
            return df
            
        def make_index_col_left(df):
            """
            Make sure order as expected by putting page col left
            """
            id_col_name = 'Page'
            cols = df.columns.tolist()
            cols.remove(id_col_name)
            
            df = df[ [id_col_name] + cols]
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
        
        idx = pd.date_range(earliest,latest) #!!!!!! fro now doing daily. When doing weekly also keep with default freq='D' . If change to 'W' alignment gets messed up. Just do daily 'D', then later can correct easily.
        OUT_OF_RANGE_FILL_VALUE = np.NaN #0 #-1 #puttign as nan casts to float and cannot convert to int


        #Do aggregation from DAILY --> WEEKLY before doing any kind of imputation
        if sampling_period=='weekly':
            AGGREGATION_TYPE = 'median'
            df = aggregate_to_weekly(df, AGGREGATION_TYPE)    

    
        #Some id's [15,16] have their missing values recorded as "-1"
        #vs. later id's have their missing values simply missing from the original csv
        #So for those id's that actually have -1, convert to NAN first:
        df.replace(-1.,np.nan,inplace=True)
    
    
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
            
            print(i,u, 'of {}'.format(unique_ids[-1]))
            if imputation_method=='lagKmedian':
                if sampling_period=='daily':
                    N_seasons = 4
                    seasonality = 7
                elif sampling_period=='weekly':
                    N_seasons = 4
                    seasonality = 1
                dd = imputation_lagKmedian_single_series(dd,seasonality,N_seasons,OUT_OF_RANGE_FILL_VALUE)

            #Data augmentation
            dd = data_augmentation(dd)
            
            df_list.append(dd)
        
        df = pd.concat(df_list,axis=0)
        #cols = df.columns.tolist()
        #df = df[cols[-1:]+cols[:-1]]
        df.reset_index(drop=True,inplace=True)
        
        
        #If we did aggregation, then above reogranization will have many of the columns Nan / -1,
        #since e.g. went from daily to weekly, then 6 days of the week will look empty. So remove them.
        if sampling_period=='weekly':
            AGGREGATION_TYPE = 'median'
            df = remove_downsample_columns(df, OUT_OF_RANGE_FILL_VALUE)
            
        
        
        
        # =============================================================================
        # Just for analysis: look at kinds of gaps in series, for DAILY data
        # =============================================================================
        #VERBOSE = False
        #if VERBOSE:
        #    __missing_vals_distribution(df,OUT_OF_RANGE_FILL_VALUE)     
            

        
        
        #Imputation, dealing with missing seasonality blocks / out of phase
        if imputation_method=='median' or imputation_method=='mean':
            df = do_imputation(df,imputation_method)
            #Could do impoutation then downsampling, vs. downsampling then imputation ... unclear which is better here in general.
            #for now assume we do ipmutation THEN aggregation:
            #df = aggregate(df,sampling_period)


        #Reorder some things just in case
        df = make_index_col_left(df)
        print(df)

        #SHould end up with a csv that is rows are series (each id), cols are dates
        #:eftmost col should be "Pages" to be same as Kaggle format
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
    #save_path = os.path.join(os.path.split(myDataDir)[0],f"train_2[ours_{sampling_period}].csv")
    save_path = os.path.join(os.path.split(myDataDir)[0],"train_2_ours_{}.csv".format(sampling_period))
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
    IMPUTATION_METHOD = 'lagKmedian' #'median' #'STL' #'lagKmedian' #None
    START_DATE = '2015-01-01' #None
    END_DATE = '2017-12-31' #None
    REMOVE_ID_LIST = []#[3,4]#id's for locations that are no longer useful
    SAMPLING_PERIOD = 'daily' #'daily', 'weekly', 'monthly'
    RANDOM_SEED = None

    # =============================================================================
    #     MAIN
    # =============================================================================
    print('START_DATE',START_DATE)
    print('END_DATE',END_DATE)
    print('REMOVE_ID_LIST',REMOVE_ID_LIST)
    print('IMPUTATION_METHOD',IMPUTATION_METHOD)
    print('myDataDir',myDataDir)
    print('SAMPLING_PERIOD',SAMPLING_PERIOD)
    
    #Seed random number generator in case of doing data augmentation:
    np.random.seed(RANDOM_SEED)
    
    #Load
    df = load_my_data(myDataDir)
    
    #Remove any bad/irrelevant cities
    df = remove_cities(df,REMOVE_ID_LIST)
    
    #Put into same format as used by Kaggle, save out csv's    
    df = format_like_Kaggle(df, myDataDir, IMPUTATION_METHOD, SAMPLING_PERIOD, start_date=START_DATE, end_date=END_DATE)

