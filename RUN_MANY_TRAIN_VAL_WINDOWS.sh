#chmod 777 RUN_MANY_TRAIN_VAL_WINDOWS.sh
#./RUN_MANY_TRAIN_VAL_WINDOWS.sh
#Run over many history_window_size - horizon_window_size length pairs
#Compile results, analyze performance as (2D) heatmap

#At this point, models have been trained already. Trained by randomizing over
#range of history and horizon sizes [~train,validation phases].
#Now hopefully the models are reasonably good across a range of values of 
#history/horizon lengths. 
#Now, assess performance (walk-forward SMAPE on test set) as a function of 
#(fixed) history and horizon sizes.
#I.e. during training phase, the history and horizon are random variables that 
#change randomly for every step of every batch. Vs. during inference, use 
#fixed settings of history and horizon sizes and get an SMAPE value, then 
#change the fixed history/horizon parameters and get another SMAPE value, etc.,
#over a range of histories/horizons. This way we can see if the model does well 
#on short series also. Of course we expect that as history->infinity and 
#horizon->1, error will decrease.


#HISTORY_SIZES="1 2 5 10 20 50 100 150 200 250 300"
#HORIZON_SIZES="1 2 5 10 20 50 100"
#e.g. HISTORY_SIZES has NAN SMAPE -> 2 problem with as big as size 50

HISTORY_SIZES="100 150"
HORIZON_SIZES="33 66"
#just to test...
MAX_EPOCH=2


#One time clean up
cd data
rm -R vars/
rm -R cpt/
rm -R cpt_tmp/
rm -R logs/
rm *.pkl
cd ..
#ls -l data/




#Now that all training is done, can run predictions
#python3 PREDICT.py !!!!!make window sizes as params
for v in $HORIZON_SIZES; do
    #Clea up between feature sets
    cd data
    rm -R vars/
    rm -R cpt_tmp/
    rm *.pkl
    cd ..
    #Create the features for our data
    echo 'running make_features.py with --add_days='$v
    python3 make_features.py data/vars ours daily full --add_days=$v
    for t in $HISTORY_SIZES; do
        echo 'history window = '$t 'horizon window = '$v
        echo 'running trainer.py'
        NAME="val$v-train$t"
        echo 'NAAME='$NAME
        python3 trainer.py full daily --name $NAME --hparam_set=s32 --n_models=3 --asgd_decay=0.99 --max_steps=11500 --save_from_step=10 --max_epoch=$MAX_EPOCH --patience=5 --verbose --horizon_window_size=$v --history_window_size=$t
    done
done



#from trainer.py, when have save_epochs_performance==True:
#format of saved "{logdir}/{name}_epochs_performance.np" numpy array is:
#2D array, dims = [epochs, 9]
#where epochs is number of epochs that successfully completed (<max_epochs if there was early stopping)
#9 is because 9 metrics are tracked. They are ordered as:
#output_list.append([eval_mae.best_epoch, eval_mae_side.best_epoch, eval_smape.best_epoch, eval_smape_side.best_epoch,
#           eval_mae.avg_epoch,  eval_mae_side.avg_epoch,  eval_smape.avg_epoch,  eval_smape_side.avg_epoch,
#           trainer.has_active()])
#For overall performance assessment, average together the last T=2or3 epochs SMAPE values 





# ==============================================================================
# now make heatmaps of performance:
# ==============================================================================
echo 'Making performance heatmaps'
python3 PERFORMANCE_HEATMAPS.py full daily --name $NAME
