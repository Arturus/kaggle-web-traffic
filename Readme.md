# RNN-based Encoder-Decoder for Time Series Forecasting w/ Quantiles 


Based on Arturus'
Kaggle Web Traffic Time Series Forecasting
1st place solution
https://github.com/Arturus/kaggle-web-traffic
![predictions](images/predictions.png)
See also [detailed model description](how_it_works.md)

-----------------------------------

GK modifications for own forecasting application:

1) Architecture improvements:
	- Recursive feedforward postprocessor: after getting sequence of predictions from RNN-based decoder, refine predictions in 2 layer MLP using nearby timesteps predictions + features + context.
	- give encoded representation vector as context to every decoder timestep
	- K step lookback: ideally the RNN would learn a hidden state representation that ~completely describes state of the system. In realiy, that may be too much to expect. In addition to previous timestep prediction y_i-1, also feed in y_i-2,...,y_i-K for K-step lookback. [~same as using lagged features]
2) Performance Analysis:
	- performance analysis of test set SMAPE as function of history/horizon window sizes [randomized uniformly in training over all min-max range of history/horizon window sizes]
	- 
2) More features, relevant to my data. More focus on seasonalities, and "spiral encoding" for holidays. Automated data augmentation.
3) Dealing with holes/sparsity as in my data.



The complete pipeline is:

1. $source activate gktf.               #previously set up a conda environment w/ Python 3.6, tensorflow 1.4.0, to match same versions as Kaggle solution
2. $cd ..../kaggle-web-traffic
3. $python PREPROCESS.py               #Maximize reuse of existing architecture: just put my data in exact same format as Kaggle competition csv's
4. $./MAKEFEATURES_TRAIN_ALL.sh         #For backtestign in chunks method [4 partially overlapping train-test set pairs]
5. $python RUN_ALL_PREDICTIONS.py      #Run predictions for every ID over triplets of (history, horizon, start point)
6. $python PERFORMANCE_HEATMAPS.py     #Analyze the prediction metrics across different dimensions 
7. $python quantile_plots.py			#FOr a subet of the predictions, get an idea what the quantiles look like





---------------------------------------
#Just in case making new features
cd data
rm -R vars*
rm -R cpt/
rm -R cpt_tmp/
rm -R logs/
rm *.pkl
cd ..
ll data/

python3 make_features.py data/TRAINset1 ours daily full --add_days=0
python3 make_features.py data/TESTset1 ours daily full --add_days=0

#For backtesting in 4 chunks, no longer do this. Run the script MAKEFEATURES_TRAIN_ALL.py to automate feature making and training all 4 chunks.
python3 trainer.py full daily --name=TRAINset1 --hparam_set=encdec --n_models=3 --asgd_decay=0.99 --max_steps=11500 --save_from_step=3 --patience=5 --max_epoch=50 --save_epochs_performance


----------------------------------------------------------------------------------------------------------------------------------------------------------
To do:
2. for weekly. monthly inputs, need to change few places in tensorflow code
3. Prediction intervals
4. Architecture improvements: bi enc, dilated; randomly dilated; randomly dilated with bounds per layer
4. MLP direct multihorizon
5. custom attention [e.g. position specific]
6. VAE aug