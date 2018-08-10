#WHen doing the chunking backtest approach, need to train/retrain model after 
#each new chunk of training data comes in.

#For this setup, just retrain from scrach (not starting at last checkpoint of 
#previous training chunk; completely starting over again)


# ==============================================================================
# PARAMETERS
# ==============================================================================
#For each of the N training sets: train model
#true false whether to remake feature sets  vs. just skip directly to training
MAKE_FEATURESETS=false
#Make some cached features for all the training/test sets
makefeats_names="TRAINset1 TRAINset2 TRAINset3 TRAINset4 TESTset1 TESTset2 TESTset3 TESTset4"
train_names="TRAINset1 TRAINset2 TRAINset3 TRAINset4"
#In training, max number of epochs to do. By 25-50 things have usually plateaud
MAX_EPOCH=50




if $MAKE_FEATURESETS; then

    echo 'Cleaning up, then remaking feature sets'
    #Clean up between feature sets
    cd data
    rm -R TRAIN*
    rm -R TEST*
    rm -R cpt/
    rm -R cpt_tmp/
    rm -R logs/
    rm *.pkl
    cd ..
    ll data/
    
        
    # =============================================================================
    # make_features.py
    # =============================================================================
    for v in $makefeats_names; do
        #Create the features for our data
        echo 'running make_features.py'
        echo $v
        python3 make_features.py data/$v ours daily full --add_days=0
    done
fi
    
    
# =============================================================================
# trainer.py    
# ============================================================================= 
for v in $train_names; do
    echo 'running trainer.py'
    echo $v
    #By default, is already doing forward split, so also do side split
    python3 trainer.py full daily --name=$v --hparam_set='encdec' --n_models=3 --asgd_decay=0.99 --max_steps=11500 --save_from_step=10 --max_epoch=$MAX_EPOCH --patience=5 --verbose --save_epochs_performance
    # --side_split    #using the side_split option gives unrealistic values for SMAPE: 
    #says training, side split, and forward step SMAPEs are all only 3-8 %, so clearly unrealistic. 
    #Not sure if Kaggle guy calculated things differently when doing side_eval option??? Just leave off for now, only do forward eval.
done