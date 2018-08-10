#Analyze the different performance metrics 
#Make the performance heatmaps

#There will be 4 different TRAIN-TEST sets, 
#each has a model trained on that train set and tested on that test set.
#So asssume to simulate production environment where we would retrain model 
#every so often, we have e.g. 4 tests of the model, each with say 3 months more 
#data appended to it. So, assume we will just do 4 separate analyses.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
import pickle
from collections import defaultdict


# =============================================================================
# PARAMETERS
# =============================================================================
OUTDIR = 'output'




# =============================================================================
# MAIN
# =============================================================================

def load_dict(path):
    with open(path,'rb') as gg:
        d = pickle.load(gg)
#    print(d)
    return d
    
    
def aggregate__overall(data_dict, real_only, id_hold_out_list, bad_ids):
    """
    For each (history,horizon) pair, marginalized over all id's and dates
    
    format is (history,horizon,backoffset,id) : {'SMAPE':smape, 'bias':bi, #'MAE':mae, 'predict_start_date':dates[0], 'predict_end_date':dates[-1]}
    """
#    print(data_dict.items())
    agg_dict = defaultdict(lambda:[])
    for k,v in data_dict.items():
        series_id = k[3]
        #Only use the real series, ignore the synthetic ones
        #(synthetic series have name like {id}__... )
        if real_only:
            if '__' in series_id:
                continue
        #If have a set of holdout id's:
        if id_hold_out_list:
            if series_id not in id_hold_out_list:
                continue
        #Regardless of mode, if this is one of the corrupted time series, ignore it:
        if series_id in bad_ids:
            continue
    
        history = k[0]
        horizon = k[1]
        smape = v['SMAPE']
        agg_dict[(history,horizon)] += [smape]        
    
    
    
    #Now get mean SMAPE
    metrics_dict = {}
    for k,v in agg_dict.items():
        mean = np.nanmean(v)
        median = np.nanmedian(v)
        sd = np.nanstd(v)
        pctl_5 = np.percentile([i for i in v if np.isfinite(i)],5)#nanpercentile
        pctl_95 = np.percentile([i for i in v if np.isfinite(i)],95)
        metrics_dict[k] = {'mean':mean, 'median':median, 'sd':sd, '5pctl':pctl_5, '95pctl':pctl_95}
    
    histories = list(np.unique([i[0] for i in metrics_dict.keys()]))
    horizons = list(np.unique([i[1] for i in metrics_dict.keys()]))
#    print(metrics_dict)
#    print(histories)
#    print(horizons)
    
    metrics_arrays = {}
    for metric in ['mean','median']:
        _array = np.nan * np.ones((len(histories),len(horizons)))
        for k,v in metrics_dict.items():
            i = histories.index(k[0])
            j = horizons.index(k[1])
            _array[i,j] = v[metric]
        metrics_arrays[metric] = _array
    print(metrics_arrays)
    return metrics_dict, histories, horizons, metrics_arrays
    



def make_heatmap(metrics_arrays, histories, horizons, outdir, name):
    """
    Visualize the SMAPE values
    """
    #For scale, get highest value for heatmap.
    #Use 200 (worst possible SMAPE), vs.
    #to improve dynamic range use the highest measured SMAPE value from the heatmaps
    
#    print('metrics_arrays')
#    print(metrics_arrays)
    for k,v in metrics_arrays.items():
        
        savename = k+'_'+name
        
        vmax = np.nanmin([200.,np.nanmax(np.ceil(v))])
        
        plt.figure()
        plt.imshow(v,vmin=0.,vmax=vmax)
        plt.title(savename,fontsize=15)
        plt.colorbar()
        plt.xlabel('Horizon',fontsize=15)
        plt.ylabel('History',fontsize=15)
        plt.xticks(np.arange(len(horizons)),horizons,fontsize=15)
        plt.yticks(np.arange(len(histories)),histories,fontsize=15)
    #    plt.grid()
        savepath = os.path.join(outdir,f'history_horizon_heatmap__{savename}.png')
        plt.savefig(savepath)
    
    
    
    
if __name__=='__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--logdir', default='data/logs', help="Directory where numpy arrays of performance are")
#    parser.add_argument('--K_last', default=3, dest='K_last', help='Save out per EPOCH metrics (NOT per step, only per EPOCH')
#    args = parser.parse_args()
#    param_dict = dict(vars(args))
#
#    make_heatmaps(**param_dict)
    
    
    #for each of the 4 dicts:
    
    #Make list of id's that were held out from training, to assess transfer ability
    HOLD_OUTS = [str(i) for i in range(500)]
    
    #Some of the ID's are just bad, have multiple month long gaps from corrupted data, etc., so can ignore them
    BAD_IDs = []#['44','46','581','582','583','584']
    
    path = os.path.join(OUTDIR,'hist_horiz__all.pickle')
    data = load_dict(path)
    
    for real_only in [True,False]:
        for id_hold_out_list in [HOLD_OUTS,[]]:
            
            r = 'real' if real_only else 'realAndsynthetic'
            h = 'holdoutsOnly' if id_hold_out_list else 'allIDs'
            name = r+'_'+h
            print(name)
            
            
            metrics_dict, histories, horizons, metrics_arrays = aggregate__overall(data, real_only, id_hold_out_list, BAD_IDs)
            make_heatmap(metrics_arrays, histories, horizons, OUTDIR, name)
            
            #Save out the metrics dict
            dict_savename = os.path.join(OUTDIR,f"hist_horiz__{name}__metrics.pickle")
            with open(dict_savename, "wb") as outp:
                pickle.dump(metrics_dict, outp)    