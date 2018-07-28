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





# =============================================================================
# 
# =============================================================================
#For histories, we care most about shorter series, so sample lower numbers more densely
HISTORY_SIZES=[7,8,10,15,20,25,35,50,70,100,150,250,350]
HORIZON_SIZES=[7,8,10,15,20,25,35,50]

# =============================================================================
# PARAMETRS
# =============================================================================
FEATURES_SET = 'full'# 'arturius' 'simple' 'full'
SAMPLING_PERIOD = 'daily'
DATA_TYPE = 'ours' #'kaggle' #'ours'
Nmodels = 3
PARAM_SETTING = 's32' #Which of the parameter settings to use [s32 is the default Kaggle one, with a few thigns modified as I want]
PARAM_SETTING_FULL_NAME = hparams.params_s32 #Which of the parameter settings to use corresponding to the PARAM_SETTING. The mapping is defined in hparams.py at the end in "sets = {'s32':params_s32,..."
OUTPUT_DIR = 'output'

RETURN_X = True
SAVE_PREDICTIONS = True











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
    """
    return np.sum(true - pred) / np.sum(true + pred)


    

def do_predictions_one_setting(history,horizon,backoffset,save_predictions,return_x):
    
    # =============================================================================
    # 
    # =============================================================================
    #read_all funcion loads the (hardcoded) file "data/all.pkl", or otherwise train2.csv
    print('loading data...')
    from make_features import read_all
    df_all = read_all(DATA_TYPE,SAMPLING_PERIOD)
    print('df_all.columns')
    print(df_all.columns)
    
    
    # =============================================================================
    # 
    # =============================================================================
    prev = df_all#.loc[:,:'2017-07-08']
    paths = [p for p in tf.train.get_checkpoint_state(f'data/cpt/{PARAM_SETTING}').all_model_checkpoint_paths]
    
    #tf.reset_default_graph()
    #preds = predict(paths, default_hparams(), back_offset=0,
    #                    n_models=3, target_model=0, seed=2, batch_size=2048, asgd=True)
    t_preds = []
    x_true = []
    for tm in range(Nmodels):
        tf.reset_default_graph()
        _ = predict(FEATURES_SET, SAMPLING_PERIOD, paths, build_hparams(PARAM_SETTING_FULL_NAME), history_window_size, horizon_window_size, back_offset=backoffset, return_x=return_x,
                        n_models=Nmodels, target_model=tm, seed=2, batch_size=2048, asgd=True)        
        if return_x:
            t_preds.append(_[0])
            x_true.append(_[1])
        else:
            t_preds.append(_)
    #def predict(features_set, sampling_period, checkpoints, hparams, history_window_size, horizon_window_size, return_x=False, verbose=False, back_offset=0, n_models=1,
    #            target_model=0, asgd=False, seed=1, batch_size=1024): #For predict: allow horizon_window_size to be fixed
    
    
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
    
    
    
    
    print(f_preds)
    
    # =============================================================================
    # save out all predictions all days (for our stuff will be relevant, for his Kaggle maybe just needed one day)
    # =============================================================================
    #firstK = 1000 #for size issues, for now while dev, just a few to look at
    #ggg = f_preds.iloc[:firstK]
    #ggg.to_csv('data/all_days_submission.csv.gz', compression='gzip', index=False, header=True)
    if save_predictions:
        f_preds.to_csv(f'{OUTPUT_DIR}/all_predictions_ours.csv.gz', compression='gzip', index=False, header=True)
    
    
    
    
    # =============================================================================
    # visualize to do wuick check
    # =============================================================================
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
        pathname = os.path.join(OUTPUT_DIR, 'fig_{}.png'.format(jj))
        plt.savefig(pathname)
        plt.close()
        
        
        
    #Cannot view on the AWS so move to local:   
    #zip -r output.zip output
    #cp output.zip /home/...../sync
    return preds, x_true
    
    
    
    
    
if __name__ == '__main__':
    for history in HISTORY_SIZES:
        for horizon in HORIZON_SIZES:
            print('HISTORY ',history, 'of ', HISTORY_SIZES)
            print('HORIZON ',horizon, 'of ', HORIZON_SIZES)
            #Get the range of values that will step through for 
            offs = [i for i in range(data_timesteps-history,horizon+1)]
            for backoffset in offs:
                print('backoffset ',backoffset, 'of ', offs)
                preds, x_true = do_predictions_one_setting(history,horizon,backoffset,SAVE_PREDICTIONS,RETURN_X)
                print(x_true)
                print(preds)
                x=yyyyyyyy                
                print('\n'*10)