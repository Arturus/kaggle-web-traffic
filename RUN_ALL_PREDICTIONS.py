import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams

from make_features import read_all




# =============================================================================
# 
# =============================================================================
#For histories, we care most about shorter series, so sample lower numbers more densely
HISTORY_SIZES=[7,350]#[7,8,10,15,20,25,35,50,70,100,150,250,350]
HORIZON_SIZES=[7,60]#[7,10,20,30,40,50,60]
EVAL_STEP_SIZE=4#step size for evaluation. 1 means use every single day as a FCT to evaluate on. E.g. 3 means step forward 3 timesteps between each FCT to evaluate on.

# =============================================================================
# PARAMETRS
# =============================================================================
TEST_DF_PATH = r"data/train_2_ours_daily_TEST.csv"
TEST_dir = r"data/vars_TEST"
FEATURES_SET = 'full'# 'arturius' 'simple' 'full'
SAMPLING_PERIOD = 'daily'
DATA_TYPE = 'ours' #'kaggle' #'ours'
Nmodels = 3
PARAM_SETTING = 's32' #Which of the parameter settings to use [s32 is the default Kaggle one, with a few thigns modified as I want]
PARAM_SETTING_FULL_NAME = hparams.params_s32 #Which of the parameter settings to use corresponding to the PARAM_SETTING. The mapping is defined in hparams.py at the end in "sets = {'s32':params_s32,..."
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

def mean_bias(true, pred):
    """
    Check if the forecasts are biased up or down
    """
    return np.mean(np.sum(true - pred) / np.sum(true + pred))


    
def do_predictions_one_setting(history,horizon,backoffset,TEST_dir,save_plots,n_series):
    
    # =============================================================================
    # 
    # =============================================================================
    #read_all funcion loads the (hardcoded) file "data/all.pkl", or otherwise train2.csv
    print('loading data...')

    df_all = read_all(DATA_TYPE,SAMPLING_PERIOD,'test')
    print('df_all.columns')
    print(df_all.columns)
#        filename = f'train_2_{data_type}_{sampling_period}'
#        df = read_file(filename)    
    

    batchsize = n_series #For simplicity, just do all series at once if not too many for memory
    # =============================================================================
    # 
    # =============================================================================
    prev = df_all#.loc[:,:'2017-07-08']
    paths = [p for p in tf.train.get_checkpoint_state(f'data/cpt/{PARAM_SETTING}').all_model_checkpoint_paths]
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
    


if __name__ == '__main__':
    
    print('TEST_DF_PATH',TEST_DF_PATH)

    groundtruth = pd.read_csv(TEST_DF_PATH)
    groundtruth.sort_values(['Page'])    
    
    data_timesteps, N_series = get_data_timesteps_Nseries(TEST_DF_PATH)
    
    hist_horiz__all = {}
#    hist_horiz__real_only = {}
#    hist_horiz__dayofweek = {}
#    hist_horiz__holidays = {}
    for history in HISTORY_SIZES:
        for horizon in HORIZON_SIZES:
            print('HISTORY ',history, 'of ', HISTORY_SIZES)
            print('HORIZON ',horizon, 'of ', HORIZON_SIZES)
            #Get the range of values that will step through for 
            offs = [i for i in range(horizon, data_timesteps - history +1, EVAL_STEP_SIZE)]
            
            
            for backoffset in offs:
                print('backoffset ',backoffset, 'of ', offs)
                f_preds = do_predictions_one_setting(history,horizon,backoffset,TEST_dir,SAVE_PLOTS,N_series)
                print(f_preds)
                
                dates = f_preds.columns
                print(dates)
                
                #For each series
                inds = f_preds.index.values
                for i, series in enumerate(f_preds):
                    _id = inds[i]
                    true = groundtruth.loc[groundtruth['Page']==_id]
                    true = true.loc[true.isin(groundtruth)]
                    print(true)
                    #Get smape, mae, bias over this prediction
                    smape = mean_smape(true, series.values)
#                    mae = asdasdasd
                    bias = mean_bias(true, series.values)                    
                    hist_horiz__all[(history,horizon,backoffset,seris)] = {'SMAPE':smape, 'MAE':0, 'bias':bias}
                

                
                #Care about the metrics within different partitions:
                #Beside just history and horizon size, also consider:
                #real vs. synthetic augmented series
                #training ID vs. new ID only in TEST set
                #series contains holiday vs. only non-holidays
                #day of week
                
                
                
                print('f_preds',f_preds)
                print('\n'*10)