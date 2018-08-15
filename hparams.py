import tensorflow.contrib.training as training
#import re

# Manually selected params
params_encdec = dict(
    batch_size=123,#256,
    #train_window=380,
#    train_window=283,#now make this a bash input to do train-validation window size performance heatmaps
    #train_window=30,#try 65 w our data to see if allows more samples through filter
    train_skip_first=0,
    rnn_depth=267,
    use_attn=False,
    attention_depth=64,
    attention_heads=1,
    encoder_readout_dropout=0.4768781146510798,

    encoder_rnn_layers=3,
    decoder_rnn_layers=1,

    # decoder_state_dropout_type=['outside','outside'],
    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[0.975, 1.0, 1.0],  # min 0.95
    decoder_state_dropout=[0.99, 0.995, 0.995],  # min 0.95
    decoder_variational_dropout=[False, False, False],
    # decoder_candidate_l2=[0.0, 0.0],
    # decoder_gates_l2=[0.0, 0.0],
    #decoder_state_dropout_type='outside',
    #decoder_input_dropout=1.0,
    #decoder_output_dropout=1.0,
    #decoder_state_dropout=0.995, #0.98,  # min 0.95
    # decoder_variational_dropout=False,
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,

    fingerprint_fc_dropout=0.8232342370695286,
    gate_dropout=0.9967589439360334,#0.9786,
    gate_activation='none',
    encoder_dropout=0.030490422531402273,
    encoder_stability_loss=0.0,  # max 100
    encoder_activation_loss=1e-06, # max 0.001
    decoder_stability_loss=0.0, # max 100
    decoder_activation_loss=5e-06,  # max 0.001
    
    
    
    # =============================================================================
    # RANDOMIZING OVER WINDOW SIZES (in training only)
    # =============================================================================
    #Instead of fixed size windows, do training phase over range of window sizes
    #drawn uniformly from [a,b]. Another form of randomization/regularization, 
    #but more importantly this way model can generalize to different lengths so
    #we can more fairly assess performance over range of history/horizon windows:
    history_window_size_minmax=[7,365],
    horizon_window_size_minmax=[7,60],    

    
    
    # =============================================================================
    # DECODER OPTIONS
    # =============================================================================
    
    # CONTEXT
    #Kaggle model architecture is more like a basic many-to-many RNN, not really a
    #usual encoder-decoder architecture since computational graph does not have 
    #connections from encoded representation to each decoder time step (only to 1st
    #decoder timestep). Set below to True to use encoder-decoder; set False to use
    #Kaggle architecture not really true encoder-decoder
    RECURSIVE_W_ENCODER_CONTEXT=True,

    # LAGGED FEATURES / LOOKBACK
    #Lookback K steps: [without specifying, default previous Kaggle setting is K=1]:
    #for predicting y_i, insteda of just feeding in previous K=1 prediction (y_i-1),
    #feed in all previous K predictions: y_
    LOOKBACK_K = 3, #!!!!Can NOT set this to be bigger than min history size (history_window_size_minmax[0])
    #since then depending on random draw would possibly need to look back further than history size.
    


    # =============================================================================
    # COMPLETELY DIFFERENT DECODERS    
    # =============================================================================
    # Alternative decoders. Can only do one of these (cannot have both True)
    
    # MLP POSTPROCESSOR (ADJUST PREDICTIONS IN LOCAL WINDOWS, AND CAN DO QUANTILES)
    #True or False to use MLP module postprocessor to locally adjust estimates
    DO_MLP_POSTPROCESS=True,#True,#False
    MLP_POSTPROCESS__KERNEL_SIZE=15,
    MLP_POSTPROCESS__KERNEL_OFFSET=7,
    
    
    # DIRECT MLP DECODER (REPLACE RNN CELLS IN DECODER WITH MLP MODULES, AND DO QUANTILES)
    #Do a direct, quantile forecast by using an MLP as decoder module instead of RNN/LSTM/GRU cells:
    MLP_DIRECT_DECODER=False,
    LOCAL_CONTEXT_SIZE=8,
    GLOBAL_CONTEXT_SIZE=64,
    
    
    
    # QUANTILE REGRESSION
    # For whatever kind of decoder, whether or not to use quantiles
    DO_QUANTILES=False,
    #If doing quantile regression in addition to point estimates trained to minimize SMAPE.
    #Also, since SMAPE point estimates are biased positive, can use alternative
    #point estimator trainde by pinball loss on quantiles < 50 [e.g. 45,38, etc., see what has bias ~0].
    #So if doing quantiles, no longer optimizing SMAPE, but report it anyway to see. So, use the 0th element of QUANTILES list is used as the point estimate for SMAPE
    #(but SMAPE will not be used in loss function: instead will use the average quantile loss (ave over all quantiles))
    #If not using quantile regression, list is ignored
    QUANTILES = [.45, .47, .5]#.05, .25, .40, .50, .60, 75, .95]
    
    
    #Losses summed together using lembda weighting. 
    #Vs. if False, just directly optimize quantile loss and ignore SMAPE [but still look at it for TEST sets]
    #LAMBDA=.01 #Scale factor for relative weight of quantile loss for the point estimate SMAPE loss. Only relevant if SMAPE_AND_QUANTILE=True
)





def_params = params_encdec

sets = {
    'encdec':params_encdec,
}


def build_hparams(params=def_params):
    return training.HParams(**params)


def build_from_set(set_name):
    return build_hparams(sets[set_name])



