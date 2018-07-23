"""
Created on Mon Jun 18 14:03:35 2018

@author: gk
"""



#After training, do the predictions [but here  as a script instead of .ipynb]


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
# PARAMETRS
# =============================================================================
FEATURES_SET = 'full'# 'arturius' 'simple' 'full'
SAMPLING_PERIOD = 'daily'
DATA_TYPE = 'ours' #'kaggle' #'ours'
Nmodels = 3
PARAM_SETTING = 's32' #Which of the parameter settings to use [s32 is the default Kaggle one, with a few thigns modified as I want]
PARAM_SETTING_FULL_NAME = hparams.params_s32 #Which of the parameter settings to use corresponding to the PARAM_SETTING. The mapping is defined in hparams.py at the end in "sets = {'s32':params_s32,..."
OUTPUT_DIR = 'output'






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
for tm in range(Nmodels):
    tf.reset_default_graph()
    t_preds.append(predict(FEATURES_SET, SAMPLING_PERIOD, paths, build_hparams(PARAM_SETTING_FULL_NAME), back_offset=0, history_window_size=283, horizon_window_size=63,
                    n_models=Nmodels, target_model=tm, seed=2, batch_size=2048, asgd=True))


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
f_preds.to_csv(f'{OUTPUT_DIR}/all_predictions_ours.csv.gz', compression='gzip', index=False, header=True)




# =============================================================================
# visualize to do wuick check
# =============================================================================
"""
pages = ['(236984)_Astier_fr.wikipedia.org_all-access_all-agents', \
         '龍抬頭_zh.wikipedia.org_mobile-web_all-agents',\
         "'Tis_the_Season_(Vince_Gill_and_Olivia_Newton-John_album)_en.wikipedia.org_mobile-web_all-agents",\
         'Peter_Townsend_(RAF_officer)_en.wikipedia.org_mobile-web_all-agents',\
         "Heahmund_en.wikipedia.org_desktop_all-agents"]
"""

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
    
    
    
    
    
    
    
    
    
        
#For the Kaggle data, can also output compeition submission format:    
if DATA_TYPE=='kaggle':   
    # =============================================================================
    # load, maniupalte test data    
    # =============================================================================
    def read_keys():
        import os.path
        key_file = 'data/keys2.pkl'
        if os.path.exists(key_file):
            return pd.read_pickle(key_file)
        else:
            print('Reading keys...')
            raw_keys = pd.read_csv('data/key_2.csv.zip')
            print('Processing keys...')
            pagedate = raw_keys.Page.str.rsplit('_', expand=True, n=1).rename(columns={0:'page',1:'date_str'})
            keys = raw_keys.drop('Page', axis=1).assign(page=pagedate.page, date=pd.to_datetime(pagedate.date_str))
            del raw_keys, pagedate
            print('Pivoting keys...')
            pkeys = keys.pivot(index='page', columns='date', values='Id')
            print('Storing keys...')
            pkeys.to_pickle(key_file)
            return pkeys
    keys = read_keys()    
    
    # =============================================================================
    # 
    # =============================================================================
    subm_preds = f_preds.loc[:, '2017-09-13':]
    assert np.all(subm_preds.index == keys.index)
    assert np.all(subm_preds.columns == keys.columns)
    answers = pd.DataFrame({'Id':keys.values.flatten(), 'Visits':np.round(subm_preds).astype(np.int64).values.flatten()})
    answers.to_csv(f'{OUTPUT_DIR}/submission.csv.gz', compression='gzip', index=False, header=True)
    
    
    
    print('f_preds')
    print(f_preds)
    
    print('missing')
    print(prev.loc[missing_pages, '2016-12-15':])