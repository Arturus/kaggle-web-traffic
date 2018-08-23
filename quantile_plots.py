#Look at some example quantile predictions.
#Not fully integrated in to the full pipeline yet, this script just
#looks at 1 example test set, and for one forecast creation time.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# =============================================================================
# PARAMETERS
# =============================================================================

#Which cities to look at
IDs = [i for i in range(1200)]#[1,2,3,5,6,12,622,555]     [1,273,284, 245, 385, 458]

#The original test set data that has the true values
groundtruth_path = r".../Desktop/forecasting/exData/4splits_aug/ours_daily_augmented__TESTset1.csv"
#corrsponding to r".../Desktop/forecasting/kaggle-web-traffic/output/quantiles/testset1"











# =============================================================================
# MAIN
# =============================================================================

#each csv of predictions is named as format:
#"{quantile}__{history}_{horizon}_{chunk}.csv"

files = os.listdir('.')
files = [i for i in files if i.endswith('.csv')]
files = [i for i in files if not i.startswith('0.4')]
files = files[::-1]#reverse order for plotting so .05th on bottom, .95th on top

actual_df = pd.read_csv(groundtruth_path)

for id_ in IDs:
    _ = []
    quantiles = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
#        print(df)
        row = df.loc[df['Unnamed: 0']==id_].head(1) #in case predictions repeated
#        print(row)
        _.append(row.values[:,1:].flatten())
        quantiles.append(os.path.split(f)[1][:-4])
#    _=np.array(_)
#    print(quantiles)
    dates = df.columns.tolist()[1:]
#    print(dates)
    try:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        plt.title('{}'.format(id_))
        colors = ['r','g','b','g','r']
        lines = ['--','--','-','--','--']
        for qq in range(len(_)):
            plt.plot(_[qq], color=colors[qq], linestyle=lines[qq], label=quantiles[qq])
        x = np.arange(len(_[0]))
    #    print(_[0])
        ax.fill_between(x,_[0],_[-1],color='r',alpha=.2)
        ax.fill_between(x,_[1],_[-2],color='g',alpha=.2)
    #    print(_[1])
        groundtruth = actual_df.loc[actual_df['Page']==id_]
        groundtruth = groundtruth[dates].T
        plt.plot(groundtruth,color='k',label='actual')
        K=14
        plt.xticks(np.arange(len(dates))[::K],dates[::K])
        plt.legend(numpoints=1)
    #        plt.show()
        plt.savefig('{}.png'.format(id_))
    except:
        continue