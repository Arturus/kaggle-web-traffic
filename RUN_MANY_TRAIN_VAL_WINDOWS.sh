#chmod 777 RUN_MANY_TRAIN_VAL_WINDOWS.sh
#./RUN_MANY_TRAIN_VAL_WINDOWS.sh
#Run over many train_window - predict_window length pairs
#Compile results, analyze performance as (2D) heatmap


#TRAIN_WINDOWS="1 2 5 10 20 50 100 150 200 250 300"
#VALIDATION_WINDOWS="1 2 5 10 20 50 100"
#e.g. TRAIN_WINDOWS has NAN SMAPE -> 2 problem with as big as size 50

TRAIN_WINDOWS="100 150"
VALIDATION_WINDOWS="33 66"
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


for v in $VALIDATION_WINDOWS; do
    #Clea up between feature sets
    cd data
    rm -R vars/
    rm -R cpt_tmp/
    rm *.pkl
    cd ..
    #Create the features for our data
    echo 'running make_features.py with --add_days='$v
    python3 make_features.py data/vars ours daily full --add_days=$v
    for t in $TRAIN_WINDOWS; do
        echo 'validation window = '$v 'validation window = '$t
        echo 'running trainer.py'
        NAME="val$v-train$t"
        echo 'NAAME='$NAME
        python3 trainer.py full daily --name $NAME --hparam_set=s32 --n_models=3 --asgd_decay=0.99 --max_steps=11500 --save_from_step=10 --max_epoch=$MAX_EPOCH --patience=5 --verbose --predict_window=$v --train_window=$t
    done
done



#Now that all training is done, can run predictions
#python3 PREDICT.py !!!!!make window sizes as params

#now make heatmaps of performance: