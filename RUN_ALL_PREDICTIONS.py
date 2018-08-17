import tensorflow as tf

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams

from make_features import read_all

import pickle
import time
from pandas import ExcelWriter


# =============================================================================
# PARAMETRS
# =============================================================================
#For histories, we care most about shorter series, so sample lower numbers more densely
HISTORY_SIZES=[7,8,10,12,15,20,30,50,100]#,200,360]
HORIZON_SIZES=[7,8,10,12,15,20,30,60]
EVAL_STEP_SIZE=4#step size for evaluation. 1 means use every single day as a FCT to evaluate on. E.g. 3 means step forward 3 timesteps between each FCT to evaluate on.
PREDICT_MODE = 'backtest'#'disjoint'
NAMES = ['TESTset1', 'TESTset2', 'TESTset3', 'TESTset4']

FEATURES_SET = 'full'# 'arturius' 'simple' 'full'
SAMPLING_PERIOD = 'daily'
DATA_TYPE = 'ours' #'kaggle' #'ours'
Nmodels = 3
PARAM_SETTING = 'encdec' #Which of the parameter settings to use [s32 is the default Kaggle one, with a few thigns modified as I want]
PARAM_SETTING_FULL_NAME = hparams.params_encdec #Which of the parameter settings to use corresponding to the PARAM_SETTING. The mapping is defined in hparams.py at the end in "sets = {'s32':params_s32,..."
OUTPUT_DIR = 'output'

SAVE_PLOTS = False












# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# Performance Metrics
# =============================================================================
def smape(true, pred):
    summ = np.abs(true) + np.abs(pred)
    smape = np.where(summ == 0, 0, np.abs(true - pred) / summ)
    #return np.mean(kaggle_smape) * 200
    return smape * 200

def mean_smape(true, pred):
    raw_smape = smape(true, pred)
    masked_smape = np.ma.array(raw_smape, mask=np.isnan(raw_smape))
    return masked_smape.mean()

def bias(true, pred):
    """
    Check if the forecasts are biased up or down
    
    All of the predictions have already been clipped to 0 min.
    Actual is always nonnegative (and 0 means missing so can mask)
    So if pred+true is 0, means missing, can ignore those
    """
    summ = pred + true
    bias = np.where(summ == 0, 0, (pred - true) / summ)
    return 100. * bias

def mean_bias(true, pred):
    raw_bias = bias(true, pred)
    masked_bias = np.ma.array(raw_bias, mask=np.isnan(raw_bias))
    return raw_bias.mean()


    
def do_predictions_one_setting(history,horizon,backoffset,TEST_dir,save_plots,n_series,chunk):
    
    # =============================================================================
    # 
    # =============================================================================
    #read_all funcion loads the (hardcoded) file "data/all.pkl", or otherwise train2.csv
    print('loading data...')

    df_all = read_all(DATA_TYPE,SAMPLING_PERIOD,f'TEST{chunk}')
    print('df_all.columns')
    print(df_all.columns)
#        filename = f'train_2_{data_type}_{sampling_period}'
#        df = read_file(filename)    
    

    batchsize = n_series #For simplicity, just do all series at once if not too many for memory
    print('batchsize',batchsize)
    # =============================================================================
    # 
    # =============================================================================
    prev = df_all#.loc[:,:'2017-07-08']
    paths = [p for p in tf.train.get_checkpoint_state(f'data/cpt/TRAIN{chunk}').all_model_checkpoint_paths]
    #tf.reset_default_graph()
    #preds = predict(paths, default_hparams(), back_offset=0,
    #                    n_models=3, target_model=0, seed=2, batch_size=2048, asgd=True)
    t_preds = []
    for tm in range(Nmodels):
        tf.reset_default_graph()
        _ = predict(FEATURES_SET, SAMPLING_PERIOD, paths, TEST_dir, build_hparams(PARAM_SETTING_FULL_NAME), history, horizon, back_offset=backoffset, return_x=False,
                    n_models=Nmodels, target_model=tm, seed=2, batch_size=batchsize, asgd=True)
        t_preds.append(_)
        
    
    # =============================================================================
    # average the N models predictions
    # =============================================================================
    preds = sum(t_preds)/float(Nmodels)
    
    
    
    # =============================================================================
    # look at missing
    # =============================================================================
    missing_pages = prev.index.difference(preds.index)
    print('missing_pages',missing_pages)
    # Use zeros for missing pages
    rmdf = pd.DataFrame(index=missing_pages,
                    data=np.tile(0, (len(preds.columns),len(missing_pages))).T, columns=preds.columns)
    if DATA_TYPE=='kaggle':
        f_preds = preds.append(rmdf).sort_index()
    elif DATA_TYPE=='ours':
        f_preds = preds
    # Use zero for negative predictions
    f_preds[f_preds < 0.5] = 0
    # Rouns predictions to nearest int
    f_preds = np.round(f_preds).astype(np.int64)
    
    
    
    
#    print(f_preds)
    
    # =============================================================================
    # save out all predictions all days (for our stuff will be relevant, for his Kaggle maybe just needed one day)
    # =============================================================================
    #firstK = 1000 #for size issues, for now while dev, just a few to look at
    #ggg = f_preds.iloc[:firstK]
    #ggg.to_csv('data/all_days_submission.csv.gz', compression='gzip', index=False, header=True)
    #Instead of saving indivual, just wait and append and look at finals.
#    f_preds.to_csv(f'{OUTPUT_DIR}/all_predictions_ours.csv.gz', compression='gzip', index=False, header=True)
    
    
    
    
    # =============================================================================
    # visualize to do wuick check
    # =============================================================================
    if save_plots:
        randomK = 1000
        print('Saving figs of {} time series as checks'.format(randomK))
        pagenames = list(f_preds.index)
        pages = np.random.choice(pagenames, size=min(randomK,len(pagenames)), replace=False)
        N = pages.size
        for jj, page in enumerate(pages):
            print(f"{jj} of {N}")
            plt.figure()
            if DATA_TYPE=='kaggle':
                prev.loc[page].fillna(0).plot()#logy=True)
                f_preds.loc[page].fillna(0).plot(logy=True)
            elif DATA_TYPE=='ours':
                prev.loc[int(page)].plot()
                f_preds.loc[page].plot()
            plt.title(page)
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
            pathname = 'ddddddddddd'#os.path.join(OUTPUT_DIR, 'fig_{}.png'.format(jj))
            plt.savefig(pathname)
            plt.close()
        
        
        
    #Cannot view on the AWS so move to local:   
    #zip -r output.zip output
    #cp output.zip /home/...../sync
    return f_preds
    
    
    
def get_data_timesteps_Nseries(df_path):
    """
    Get the data_timesteps value from the TEST set data.
    Because every day will be used, it is just the number of days in the df.
    
    And get number of time series to prdict on [number of rows], to use as batchsize
    """
    df = pd.read_csv(df_path)
    columns = list(df.columns)
    columns.remove('Page')
    return len(columns), len(df)



def get_data_timesteps_Nseries__backtest(test_path,train_path):
    """
    For backtest chunk mode only.
    Get the number of data timesteps (which potentially varies per testset1,2,3,4),
    in order to determine backoffset range.
    
    Because in this mode the TEST set also includes the TRAIN ste in it, cannot 
    just use length of TEST set alone to get datatimesteps.
    """
    test = pd.read_csv(test_path)
#    test_day = test.columns[-1]
    train = pd.read_csv(train_path)
#    train_day = train.columns[-1]
    
    #Assuming consecutive days, just get diff of number columns:
    data_timesteps = len(test.columns) - len(train.columns)
    
    #Depending if did holdout id's, then TEST would have extra id's not in TRAIN
    #For batchsize, using N_series as number of rows of TEST set.
    #Since metrics are later made from dicts, if an ID is predicted on more than once,
    #is ok, since would have same key in dict and only be there once anyway.
    N_series = len(test)
    
    return data_timesteps, N_series
        
    




def SaveMultisheetXLS(list_dfs, list_sheetnames, xls_path):
    """
    xls_path - must be .xls or else will not save sheets properly
    """
    writer = ExcelWriter(xls_path)
    for s in range(len(list_dfs)):
        list_dfs[s].to_excel(writer,list_sheetnames[s],index=False)
    writer.save()
    
    
    


if __name__ == '__main__':
    
    
    
    #For the 4 chunk backtesting performance assessment
    for name in NAMES:
        
        print('name: ',name)
        chunk = name.replace('TEST','')
        TEST_DF_PATH = f"data/ours_daily_{name}.csv"
        TEST_dir = f"data/{name}"
        TRAIN_DF_PATH = TEST_DF_PATH.replace('TEST','TRAIN')
        print('TEST_DF_PATH',TEST_DF_PATH)
        print('TEST_dir',TEST_dir)
        print('TRAIN_DF_PATH',TRAIN_DF_PATH)
    
        groundtruth = pd.read_csv(TEST_DF_PATH)
        groundtruth.sort_values(['Page'])    
        print('groundtruth',groundtruth)
    
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        if PREDICT_MODE=='disjoint':
            data_timesteps, N_series = get_data_timesteps_Nseries(TEST_DF_PATH)
        elif PREDICT_MODE=='backtest':
            data_timesteps, N_series = get_data_timesteps_Nseries__backtest(TEST_DF_PATH,TRAIN_DF_PATH)
        
        hist_horiz__all = {}
        t0 = time.clock()
        for history in HISTORY_SIZES:
            for horizon in HORIZON_SIZES:
                print('HISTORY ',history, 'of ', HISTORY_SIZES)
                print('HORIZON ',horizon, 'of ', HORIZON_SIZES)
                
                #For the disjoint mode, the test set does not overlap art all w the train set. 
                #The history + horizon window must completely fit in the test set alone.
                #vs.
                #in backtest chunk mode, test set include full train set, but 
                #horizon window always starts after the train set (so horizon 
                #is fully inside test set). SO for backtest chunk mode, irrelevant
                #what history + horizon is, only matters that the horizon is fully inside TEST set.
                if (PREDICT_MODE=='disjoint') and (history+horizon >= data_timesteps):
                    print(f'history+horizon ({history+horizon}) >= data set size ({data_timesteps})')
                    continue
                if (PREDICT_MODE=='backtest') and (horizon > data_timesteps):
                    print(f'horizon ({horizon}) > test region size ({data_timesteps})')
                    continue                
                
                #Get the range of values that will step through for 
                if (PREDICT_MODE=='disjoint'):
                    offs = [i for i in range(horizon, data_timesteps - history +1, EVAL_STEP_SIZE)]
                if (PREDICT_MODE=='backtest'):
                    offs = [i for i in range(horizon, data_timesteps+1, EVAL_STEP_SIZE)]


                dflist = []
                for backoffset in offs:
                    print('backoffset ',backoffset, 'of ', offs)
                    f_preds = do_predictions_one_setting(history,horizon,backoffset,TEST_dir,SAVE_PLOTS,N_series,chunk)
                    cols = f_preds.columns
                    dates = [i.strftime('%Y-%m-%d') for i in cols]
                    print(dates)
                    
                    #For each series
                    for jj in range(len(f_preds)):
                        series = f_preds.iloc[jj]
                        _id = series.name
                        true = groundtruth[groundtruth['Page'].astype(str) ==_id]
                        
                        
                        first_pred_day = dates[0]
                        d1 = pd.date_range(first_pred_day,first_pred_day)[0] - pd.Timedelta(history,unit='D')
                        history_dates = pd.date_range(start=d1, end=first_pred_day, freq='D')[:-1]   #!!!!!! asuming daily sampling...
                        history_dates = [i.strftime('%Y-%m-%d') for i in history_dates]
                        history_missing_count = np.isnan(true[history_dates].values[0]).sum()
    #                    print('history_missing_count',history_missing_count)                    
    #                    print('true',true)
                        true = true[dates].values[0]
                        horizon_missing_count = np.isnan(true).sum()
    #                    print('horizon_missing_count',horizon_missing_count)
                        
                        #Get smape, mae, bias over this prediction
                        smp = mean_smape(true, series.values)
    #                    mae = asdasdasd
                        bi = mean_bias(true, series.values)
    #                    print(smape,bias)
                        hist_horiz__all[(history,horizon,backoffset,_id)] = {'SMAPE':smp, 
                                        'bias':bi,
                                        #'MAE':mae,
                                        'predict_start_date':dates[0],
                                        'predict_end_date':dates[-1],
                                        'history_missing_count':history_missing_count,
                                        'horizon_missing_count':horizon_missing_count
                                        }
    #                    print(hist_horiz__all)
                        
                        
                    #For saving out predictions:
                    dates = [i.strftime('%m/%d/%Y') for i in cols]
                    d = {cols[i]:dates[i] for i in range(len(cols))}
                    f_preds.rename(columns=d,inplace=True)
                    f_preds['Page'] = f_preds.index.values
                    #Depending on missing data in the test set in the history window for this backoffset,
                    #it oculd be that that particular id did not pass the train completeness threshold.
                    #Then it will not be included, but the batchsize will still be len(df), so to fill that missing
                    #id, it will repeat id's that already had predictions. THey will be identical, 
                    #so just take the 1st occurrence for those repeated id's:
                    df = []
                    u_ids = np.unique(f_preds['Page'].values)
                    for u in u_ids:
                        s = f_preds[f_preds['Page']==u]
                        if len(s)>1:
                            s = s.head(1)
                        df += [s]
                    f_preds = pd.concat(df,axis=0)
                    cols = list(f_preds.columns)
                    cols.remove('Page')
                    cols = ['Page'] + cols
                    f_preds = f_preds[cols]                 
#                    print(f_preds)
                    
                    dflist += [f_preds]
                    #Care about the metrics within different partitions:
                    #Beside just history and horizon size, also consider:
                    #real vs. synthetic augmented series
                    #training ID vs. new ID only in TEST set
                    #series contains holiday vs. only non-holidays
                    #day of week
                    
                savename = f"{str(history)}_{str(horizon)}_{name}.xls"
                savename = os.path.join(OUTPUT_DIR,savename)
                sheetnames = [str(i) for i in offs]
                SaveMultisheetXLS(dflist, sheetnames, savename)
                #each sheet is for a single backoffset, so each sheet contains all ~1800 id's
    
        
#        print(hist_horiz__all)
        t1 = time.clock()
        print('elapsed time: ',t1-t0)
        #Now that all metrics stored in dict, save dict, and analyze further
        #pickle ... hist_horiz__all
    #    print(hist_horiz__all)
        dict_savename = os.path.join(OUTPUT_DIR,f"hist_horiz__{name}.pickle")
        with open(dict_savename, "wb") as outp:
            pickle.dump(hist_horiz__all, outp)#, protocol=2)