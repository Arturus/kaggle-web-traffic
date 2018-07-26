import numpy as np
import matplotlib.pyplot as plt
import argparse





def make_heatmaps(logdir='data/logs', K_last=3):
    #Load all saved numpy arrays of performance metrics per PREDICTION run:
    all_runs = []
    eval_smapes_lastKmean = []
    array_names = [i for i in ssssss if i.endswith('epochs_performance.npy')]
    run_names = [i.split('_')[0] for i in array_names]
    for i, an in enumerate(array_names):
        x = np.load(an)
        #Get last K epoch metrics:
        j = x[-K_last:]
        eval_smapes_lastKmean.append(np.mean(j[:,5]))
        all_runs.append(x)
        
        


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='data/logs', help="Directory where numpy arrays of performance are")
    parser.add_argument('--K_last', default=3, dest='K_last', help='Save out per EPOCH metrics (NOT per step, only per EPOCH')
    args = parser.parse_args()
    param_dict = dict(vars(args))

    make_heatmaps(**param_dict)