import tensorflow as tf
from functools import partial

import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from tensorflow.python.util import nest

from cocob import COCOB
from Adam_HD_optimizer import AdamHDOptimizer
from SGDN_HD_optimizer import MomentumSGDHDOptimizer
from input_pipe import InputPipe, ModelMode


GRAD_CLIP_THRESHOLD = 10
RNN = cudnn_rnn.CudnnGRU
#RNN = tf.contrib.cudnn_rnn.CudnnLSTM
# RNN = tf.contrib.cudnn_rnn.CudnnRNNRelu





def debug_tensor_print(tensor):
    """
    Debugging mode:
        Print info about a tensor in realtime
    """
    tensor_list = [tensor.name, tf.shape(tensor), tensor]
    tensor = tf.Print(tensor, tensor_list)
    return tensor


def default_init(seed):
    # replica of tf.glorot_uniform_initializer(seed=seed)
    return layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)


def selu(x):
    """
    SELU activation
    https://arxiv.org/abs/1706.02515
    :param x:
    :return:
    """
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def cuda_params_size(cuda_model_builder):
    """
    Calculates static parameter size for CUDA RNN
    :param cuda_model_builder:
    :return:
    """
    with tf.Graph().as_default():
        cuda_model = cuda_model_builder()
        params_size_t = cuda_model.params_size()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            result = sess.run(params_size_t)
            return result


def make_encoder(time_inputs, encoder_features_depth, is_train, hparams, seed, transpose_output=True):
    """
    Builds encoder, using CUDA RNN
    :param time_inputs: Input tensor, shape [batch, time, features]
    :param encoder_features_depth: Static size for features dimension
    :param is_train:
    :param hparams:
    :param seed:
    :param transpose_output: Transform RNN output to batch-first shape
    :return:
    """
    DIRECTION = 'unidirectional'#'bidirectional',#'unidirectional', #Let's try bidirectional as well, or ,ay as well try keeping unidirectional but with order reversed, just see what happens
    def build_rnn():
        return RNN(num_layers=hparams.encoder_rnn_layers, num_units=hparams.rnn_depth,
                   input_size=encoder_features_depth,
                   direction=DIRECTION,
                   #assume merge mode default is concat??
                   #need to fix dimensions error. If could change merge mode to sum or mean or something then at least output dimension is same so might be easiest way to avoid  error ?
                   dropout=hparams.encoder_dropout if is_train else 0, seed=seed)

    static_p_size = cuda_params_size(build_rnn)
#    static_p_size = tf.Print(static_p_size,['static_p_size',static_p_size])
#    print('static_p_size',static_p_size)
    cuda_model = build_rnn()
#    time_inputs = tf.check_numerics(time_inputs,'time_inputs')
    
    params_size_t = cuda_model.params_size()
#    params_size_t = tf.Print(static_p_size,['params_size_t',params_size_t])
#    print('params_size_t',params_size_t)
    with tf.control_dependencies([tf.assert_equal(params_size_t, [static_p_size])]):
        cuda_params = tf.get_variable("cuda_rnn_params",
                                      initializer=tf.random_uniform([static_p_size], minval=-0.05, maxval=0.05,
                                                                    dtype=tf.float32, seed=seed + 1 if seed else None)
                                      )

    def build_init_state():
        batch_len = tf.shape(time_inputs)[0] #!!!!!!!! for random history/horizon size, may need to adjust
        return tf.zeros([hparams.encoder_rnn_layers, batch_len, hparams.rnn_depth], dtype=tf.float32)
#    def build_init_state():
#        batch_len = tf.shape(time_inputs)[0] #!!!!!!!! for random history/horizon size, may need to adjust
#        if DIRECTION == 'unidirectional':
#            return tf.zeros([1, hparams.encoder_rnn_layers, batch_len, hparams.rnn_depth], dtype=tf.float32)
#        if DIRECTION == 'bidirectional':
#            return tf.zeros([2, hparams.encoder_rnn_layers, batch_len, hparams.rnn_depth], dtype=tf.float32)        

    input_h = build_init_state()

    # [batch, time, features] -> [time, batch, features]
    time_first = tf.transpose(time_inputs, [1, 0, 2])
    rnn_time_input = time_first
    
    
#    cuda_params = tf.Print(cuda_params,['cuda_params',tf.shape(cuda_params),cuda_params]) #???? shape is [233892]
    
    model = partial(cuda_model, input_data=rnn_time_input, input_h=input_h, params=cuda_params)
    if RNN == tf.contrib.cudnn_rnn.CudnnLSTM:
        rnn_out, rnn_state, c_state = model(input_c=build_init_state())
    else:
        rnn_out, rnn_state = model()
        c_state = None
    if transpose_output:
        rnn_out = tf.transpose(rnn_out, [1, 0, 2])
        
    
    #Need to check for NANs that are sometimes happening
    rnn_out = tf.check_numerics(rnn_out,'rnn_out')    
    rnn_state = tf.check_numerics(rnn_state,'rnn_state')

        
#    rnn_out = tf.Print(rnn_out,['rnn_out',rnn_out])
#    rnn_state = tf.Print(rnn_state,['rnn_state',rnn_state,'encoder_features_depth',encoder_features_depth])
#    encoder_features_depth = tf.Print(encoder_features_depth,['encoder_features_depth',encoder_features_depth])
    return rnn_out, rnn_state, c_state


def compressed_readout(rnn_out, hparams, dropout, seed):
    """
    FC compression layer, reduces RNN output depth to hparams.attention_depth
    :param rnn_out:
    :param hparams:
    :param dropout:
    :param seed:
    :return:
    """
    if dropout < 1.0:
        rnn_out = tf.nn.dropout(rnn_out, dropout, seed=seed)
    return tf.layers.dense(rnn_out, hparams.attention_depth,
                           use_bias=True,
                           activation=selu,
                           kernel_initializer=layers.variance_scaling_initializer(factor=1.0, seed=seed),
                           name='compress_readout'
                           )


def make_fingerprint(x, is_train, fc_dropout, seed):
    """
    Calculates 'fingerprint' of timeseries, to feed into attention layer
    :param x:
    :param is_train:
    :param fc_dropout:
    :param seed:
    :return:
    """
    with tf.variable_scope("fingerpint"):
        # x = tf.expand_dims(x, -1)
        with tf.variable_scope('convnet', initializer=layers.variance_scaling_initializer(seed=seed)):
            c11 = tf.layers.conv1d(x, filters=16, kernel_size=7, activation=tf.nn.relu, padding='same')
            c12 = tf.layers.conv1d(c11, filters=16, kernel_size=3, activation=tf.nn.relu, padding='same')
            pool1 = tf.layers.max_pooling1d(c12, 2, 2, padding='same')
            c21 = tf.layers.conv1d(pool1, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same')
            c22 = tf.layers.conv1d(c21, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same')
            pool2 = tf.layers.max_pooling1d(c22, 2, 2, padding='same')
            c31 = tf.layers.conv1d(pool2, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same')
            c32 = tf.layers.conv1d(c31, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same')
            pool3 = tf.layers.max_pooling1d(c32, 2, 2, padding='same')
            dims = pool3.shape.dims
            pool3 = tf.reshape(pool3, [-1, dims[1].value * dims[2].value])
            if is_train and fc_dropout < 1.0:
                cnn_out = tf.nn.dropout(pool3, fc_dropout, seed=seed)
            else:
                cnn_out = pool3
        with tf.variable_scope('fc_convnet',
                               initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', seed=seed)):
            fc_encoder = tf.layers.dense(cnn_out, 512, activation=selu, name='fc_encoder')
            out_encoder = tf.layers.dense(fc_encoder, 16, activation=selu, name='out_encoder')
    return out_encoder



def attn_readout_v3(readout, attn_window, attn_heads, page_features, seed):
    # input: [n_days, batch, readout_depth]
    # [n_days, batch, readout_depth] -> [batch(readout_depth), width=n_days, channels=batch]
    readout = tf.transpose(readout, [2, 0, 1])
    # [batch(readout_depth), width, channels] -> [batch, height=1, width, channels]
    inp = readout[:, tf.newaxis, :, :]

    # attn_window = history_window_size - horizon_window_size + 1
    # [batch, attn_window * n_heads]
    filter_logits = tf.layers.dense(page_features, attn_window * attn_heads, name="attn_focus",
                                    kernel_initializer=default_init(seed)
                                    # kernel_initializer=layers.variance_scaling_initializer(uniform=True)
                                    # activation=selu,
                                    # kernel_initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
                                    )
    # [batch, attn_window * n_heads] -> [batch, attn_window, n_heads]
    filter_logits = tf.reshape(filter_logits, [-1, attn_window, attn_heads])

    # attns_max = tf.nn.softmax(filter_logits, dim=1)
    attns_max = filter_logits / tf.reduce_sum(filter_logits, axis=1, keep_dims=True)
    # [batch, attn_window, n_heads] -> [width(attn_window), channels(batch), n_heads]
    attns_max = tf.transpose(attns_max, [1, 0, 2])

    # [width(attn_window), channels(batch), n_heads] -> [height(1), width(attn_window), channels(batch), multiplier(n_heads)]
    attn_filter = attns_max[tf.newaxis, :, :, :]
    # [batch(readout_depth), height=1, width=n_days, channels=batch] -> [batch(readout_depth), height=1, width=horizon_window_size, channels=batch*n_heads]
    averaged = tf.nn.depthwise_conv2d_native(inp, attn_filter, [1, 1, 1, 1], 'VALID')
    # [batch, height=1, width=horizon_window_size, channels=readout_depth*n_neads] -> [batch(depth), horizon_window_size, batch*n_heads]
    attn_features = tf.squeeze(averaged, 1)
    # [batch(depth), horizon_window_size, batch*n_heads] -> [batch*n_heads, horizon_window_size, depth]
    attn_features = tf.transpose(attn_features, [2, 1, 0])
    # [batch * n_heads, horizon_window_size, depth] -> n_heads * [batch, horizon_window_size, depth]
    heads = [attn_features[head_no::attn_heads] for head_no in range(attn_heads)]
    # n_heads * [batch, horizon_window_size, depth] -> [batch, horizon_window_size, depth*n_heads]
    result = tf.concat(heads, axis=-1)
    # attn_diag = tf.unstack(attns_max, axis=-1)
    return result, None


def calc_smape_rounded(true, predicted, weights):
    """
    Calculates SMAPE on rounded submission values. Should be close to official SMAPE in competition
    :param true:
    :param predicted:
    :param weights: Weights mask to exclude some values
    :return:
    """
    n_valid = tf.reduce_sum(weights)
    true_o = tf.round(tf.expm1(true))
    pred_o = tf.maximum(tf.round(tf.expm1(predicted)), 0.0) #!!!!!!! for us we could even clip at 1, since 0 means measurement was missing
    summ = tf.abs(true_o) + tf.abs(pred_o)
    zeros = summ < 0.01
    raw_smape = tf.abs(pred_o - true_o) / summ * 2.0
    smape = tf.where(zeros, tf.zeros_like(summ, dtype=tf.float32), raw_smape)
    #!!!!!!!!!!! since summ is sum of absolute values of 2 rounded things, is only < .01 if is exactly = 0. For our data, this should NEVER happen, would mean unmeasured NAN, so actually this is exactly the SMAPE we want
    
#    smape = tf.Print(smape, ['pred_o',tf.shape(pred_o),pred_o, 'pred_o not round clip',tf.expm1(predicted),  'true_o',tf.shape(true_o),true_o,  'smape', smape, 'raw_smape', raw_smape])  
    return tf.reduce_sum(smape * weights) / n_valid


def smape_loss(true, predicted, weights):
    """
    Differentiable SMAPE loss
    :param true: Truth values
    :param predicted: Predicted values
    :param weights: Weights mask to exclude some values
    :return:
    """
    epsilon = 0.1  # Smoothing factor, helps SMAPE to be well-behaved near zero
    true_o = tf.expm1(true)
    pred_o = tf.expm1(predicted)
    summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = tf.abs(pred_o - true_o) / summ * 2.0
    return tf.losses.compute_weighted_loss(smape, weights, loss_collection=None)


def decode_predictions(decoder_readout, inp: InputPipe):#!!!!!quantiles
    """
    Converts normalized prediction values to log1p(pageviews), e.g. reverts normalization
    :param decoder_readout: Decoder output, shape [n_days, batch]
    :param inp: Input tensors
    :return:
    """
    # [n_days, batch] -> [batch, n_days]
    batch_readout = tf.transpose(decoder_readout) #!!!!!quantiles
    batch_std = tf.expand_dims(inp.norm_std, -1)
    batch_mean = tf.expand_dims(inp.norm_mean, -1)
    
    ret = batch_readout * batch_std + batch_mean
#    ret = tf.Print(ret, ['ret:',tf.shape(ret),ret, 'batch_readout:',batch_readout, 'batch_std:',batch_std, 'batch_mean',batch_mean])
    return ret


def quantile_loss(true, predicted, weights, quantile):
    """
    When doing quantile regression, get the pinball loss on each quantile
    """
    n_valid = tf.reduce_sum(weights)
    true_o = tf.round(tf.expm1(true))
    pred_o = tf.maximum(tf.round(tf.expm1(predicted)), 0.0)
    diff = tf.subtract(true, predicted)
    pinball = tf.reduce_mean(tf.maximum(quantile*diff, (quantile-1.)*diff))
    return tf.reduce_sum(pinball * weights) / n_valid


def calc_loss(predictions, true_y, additional_mask=None, DO_QUANTILES=False, QUANTILES=[]):
    """
    Calculates losses, ignoring NaN true values (assigning zero loss to them)
    :param predictions: Predicted values
    :param true_y: True values
    :param additional_mask:
    :return: MAE loss, differentiable SMAPE loss, competition SMAPE loss
    """
    # Take into account NaN's in true values
    mask = tf.is_finite(true_y)
    # Fill NaNs by zeros (can use any value)
    true_y = tf.where(mask, true_y, tf.zeros_like(true_y))
    # Assign zero weight to NaNs
    weights = tf.to_float(mask)
    if additional_mask is not None:
        weights = weights * tf.expand_dims(additional_mask, axis=0)

    mae_loss = tf.losses.absolute_difference(labels=true_y, predictions=predictions, weights=weights)
    
    if DO_QUANTILES:
        quantile_losses = []
        for nn, q in enumerate(QUANTILES):
            quantile_losses.extend([quantile_loss(true_y, predictions[nn+1], weights, q)])
        #For the total quantile loss, use the mean loss over all quantiles
        ave_quantile_loss = tf.reduce_mean(quantile_losses)
    else:
        ave_quantile_loss = 0.
        quantile_losses = []
        
    return mae_loss, smape_loss(true_y, predictions, weights), calc_smape_rounded(true_y, predictions,
                                                                                  weights), tf.size(true_y), ave_quantile_loss, quantile_losses


def make_train_op(loss, ema_decay=None, prefix=None):#!!!!!quantiles
#    OPTIMIZER=#'SGDN-HD',#'COCOB',#'ADAM',#'SGDN-HD',#'ADAM-HD'
#    if OPTIMIZER=='COCOB':
#        optimizer = COCOB()
#    if OPTIMIZER=='ADAM':
#        optimizer = tf.train.AdamOptimizer()
#    if OPTIMIZER=='SGD':
#        optimizer = tf.train.GradientDescentOptimizer(1e-9)
#    if OPTIMIZER=='SGDN-HD':
#        optimizer = MomentumSGDHDOptimizer()
#    if OPTIMIZER=='ADAM-HD':
#        optimizer = AdamHDOptimizer()
#    optimizer=MomentumSGDHDOptimizer(alpha_0=1e-1)#bad SMAPEs for various orders of magnitude alpha_0
    optimizer = tf.train.AdamOptimizer()
    
    glob_step = tf.train.get_global_step()

    # Add regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss + reg_losses if reg_losses else loss

    # Clip gradients
    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, GRAD_CLIP_THRESHOLD)
    sgd_op, glob_norm = optimizer.apply_gradients(zip(clipped_gradients, variables)), glob_norm

    # Apply SGD averaging
    if ema_decay:
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay, num_updates=glob_step)
        if prefix:
            # Some magic to handle multiple models trained in single graph
            ema_vars = [var for var in variables if var.name.startswith(prefix)]
        else:
            ema_vars = variables
        update_ema = ema.apply(ema_vars)
        with tf.control_dependencies([sgd_op]):
            training_op = tf.group(update_ema)
    else:
        training_op = sgd_op
        ema = None
    return training_op, glob_norm, ema


def convert_cudnn_state_v3(h_state, hparams, seed, c_state=None, dropout=1.0):
    """
    Converts RNN state tensor from cuDNN representation to TF RNNCell compatible representation.
    :param h_state: tensor [num_layers, batch_size, depth]
    :param c_state: LSTM additional state, should be same shape as h_state
    :return: TF cell representation matching RNNCell.state_size structure for compatible cell
    """

    def squeeze(seq):
        return tuple(seq) if len(seq) > 1 else seq[0]

    def wrap_dropout(structure):
        if dropout < 1.0:
            return nest.map_structure(lambda x: tf.nn.dropout(x, keep_prob=dropout, seed=seed), structure)
        else:
            return structure

    # Cases:
    # decoder_layer = encoder_layers, straight mapping
    # encoder_layers > decoder_layers: get outputs of upper encoder layers
    # encoder_layers < decoder_layers: feed encoder outputs to lower decoder layers, feed zeros to top layers
    h_layers = tf.unstack(h_state)
    
#    h_layers = tf.Print(h_layers,['h_layers',h_layers])
    
    #Regardless of relative number of layers in encoder vs. decoder, simple approach is 
    #use topmost encoder layer hidden state as the (fixed) context
    encoded_representation = wrap_dropout(h_layers[-1])
    
#    encoded_representation = tf.Print(encoded_representation,['encoded_representation',encoded_representation])
    
    #above uses a different random dropout for the "encoded representaiton" than the actual top level output.
    #This is possibly a good regularization thing since we dont expect the final hidden state to be  perfect summar/context vector,
    #so a little randomness is probably good here.
    #vs. below using topmost level exactly same dropout mask: _[-1]
    if hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers:
        _ = wrap_dropout(h_layers[hparams.encoder_rnn_layers - hparams.decoder_rnn_layers:])
        return squeeze(_), _[-1] #Use the topmost hidden state of the encoder as the encoded representaiton
#        return squeeze(_), encoded_representation #Use the topmost hidden state of the encoder as the encoded representaiton
    else:
        lower_inputs = wrap_dropout(h_layers)
        upper_inputs = [tf.zeros_like(h_layers[0]) for _ in
                        range(hparams.decoder_rnn_layers - hparams.encoder_rnn_layers)]
        return squeeze(lower_inputs + upper_inputs), lower_inputs[-1] #Use the topmost hidden state of the encoder as the encoded representaiton


def rnn_stability_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    # [time, batch, features] -> [time, batch]
    l2 = tf.sqrt(tf.reduce_sum(tf.square(rnn_output), axis=-1))
    #  [time, batch] -> []
    return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def rnn_activation_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    return tf.nn.l2_loss(rnn_output) * beta


class Model:
    def __init__(self, inp: InputPipe, hparams, is_train, seed, graph_prefix=None, asgd_decay=None, loss_mask=None):
        """
        Encoder-decoder prediction model
        :param inp: Input tensors
        :param hparams:
        :param is_train:
        :param seed:
        :param graph_prefix: Subgraph prefix for multi-model graph
        :param asgd_decay: Decay for SGD averaging
        :param loss_mask: Additional mask for losses calculation (one value for each prediction day), shape=[horizon_window_size]
        """
        self.is_train = is_train
        self.inp = inp
        self.hparams = hparams
        self.seed = seed
        self.inp = inp
        
        self.DO_MLP_POSTPROCESS = self.hparams.DO_MLP_POSTPROCESS
        self.lookback_K_actual = min(hparams.LOOKBACK_K, hparams.history_window_size_minmax[0])
        print('self.lookback_K_actual',self.lookback_K_actual)
        self.MLP_DIRECT_DECODER = self.hparams.MLP_DIRECT_DECODER
        self.DO_QUANTILES = self.hparams.DO_QUANTILES
        self.LAMBDA = self.hparams.LAMBDA
        assert not (self.MLP_DIRECT_DECODER and self.DO_MLP_POSTPROCESS), 'Cannot do both DO_MLP_POSTPROCESS and MLP_DIRECT_DECODER.\n Choose exactly 1, or neither.'
 
        


#        with tf.Graph().as_default():
#            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#            with tf.Session(config=config) as sess:
#                result = sess.run([self.inp.history_window_size, self.inp.horizon_window_size])
#                print(result)
        print('Model.py history, horizon : ', self.inp.history_window_size, self.inp.horizon_window_size)



#        inp.time_x = tf.Print(inp.time_x, ['where NANs in inp.time_x :', tf.where(tf.is_nan(inp.time_x))])
#        inp.time_x = tf.Print(inp.time_x, ['inp.time_x :', tf.shape(inp.time_x),inp.time_x])
        
        
#        inp.time_x = tf.check_numerics(inp.time_x,'inp.time_x has NANs')
#        inp.time_y = tf.Print(inp.time_y, ['inp.time_y :', tf.shape(inp.time_y),inp.time_y])
#        inp.norm_x = tf.Print(inp.norm_x, ['inp.norm_x :', tf.shape(inp.norm_x),inp.norm_x])
#        inp.time_x = tf.Print(inp.time_x, ['inp.time_x[:,:,0] :', tf.shape(inp.time_x[:,:,0]),inp.time_x[:,:,0]])
#        gggg = tf.reduce_mean(tf.cast(tf.equal(inp.norm_x, inp.time_x[:,:,0]), tf.int32))
#        inp.time_x = tf.Print(inp.time_x, ['mean tf.equal(inp.norm_x, inp.time_x[:,:,0]) should be 1 :', gggg])        
        

        encoder_output, h_state, c_state = make_encoder(inp.time_x, inp.encoder_features_depth, is_train, hparams, seed,
                                                        transpose_output=False)
        
#        h_state = tf.Print(h_state,['h_state',h_state,'encoder_output',encoder_output,'inp.time_x',inp.time_x])
        
        
        # Encoder activation losses
#        enc_stab_loss = rnn_stability_loss(encoder_output, hparams.encoder_stability_loss / tf.cast(inp.history_window_size, tf.float32)) #!!!!!!!!when random history-horizons, need to cast dtypes here
#        enc_activation_loss = rnn_activation_loss(encoder_output, hparams.encoder_activation_loss / tf.cast(inp.history_window_size, tf.float32)) #!!!!!!
        enc_stab_loss = rnn_stability_loss(encoder_output, hparams.encoder_stability_loss / inp.history_window_size) #!!!!!!!!when random history-horizons, need to cast dtypes here
        enc_activation_loss = rnn_activation_loss(encoder_output, hparams.encoder_activation_loss / inp.history_window_size) #!!!!!!
        
        
        # Convert state from cuDNN representation to TF RNNCell-compatible representation
        encoder_state, summary_z = convert_cudnn_state_v3(h_state, hparams, c_state,
#        encoder_state, summary_z = convert_cudnn_state_v3(h_state, hparams, seed, c_state,
                                               dropout=hparams.gate_dropout if is_train else 1.0)
#        encoder_state = tf.Print(encoder_state, ['encoder_state',tf.shape(encoder_state),encoder_state])
#        summary_z = tf.Print(summary_z, ['summary_z',tf.shape(summary_z),summary_z])
        
        # Attention calculations
        # Compress encoder outputs
        enc_readout = compressed_readout(encoder_output, hparams,
                                         dropout=hparams.encoder_readout_dropout if is_train else 1.0, seed=seed)
        
        # Calculate fingerprint from input features
        if hparams.use_attn:
            fingerprint_inp = tf.concat([inp.lagged_x, tf.expand_dims(inp.norm_x, -1)], axis=-1)
            fingerprint = make_fingerprint(fingerprint_inp, is_train, hparams.fingerprint_fc_dropout, seed)
            # Calculate attention vector
            attn_features, attn_weights = attn_readout_v3(enc_readout, inp.attn_window, hparams.attention_heads,
                                                      fingerprint, seed=seed)

        print('building decoder')

        # Run decoder
        #... = decoder(encoder_state, attn_features, prediction_inputs, previous_y)
        decoder_targets, decoder_outputs = self.decoder(encoder_state,
                                                        attn_features if hparams.use_attn else None,
                                                        summary_z if hparams.RECURSIVE_W_ENCODER_CONTEXT else None,
                                                        inp.time_y, inp.norm_x[:, -self.lookback_K_actual:]) #in decoder function def:   inp.time_y = "prediction_inputs";  inp.norm_x[:, -1] = "previous_y" (i.e. the final x normalizd))
        
#        decoder_targets = tf.Print(decoder_targets,['decoder_targets BEFORE',tf.shape(decoder_targets),decoder_targets,'decoder_outputs',tf.shape(decoder_outputs),decoder_outputs])
        
        
        
        #If doing the MLP postprocessing step to adjust predictions:
        if self.DO_MLP_POSTPROCESS:
            #Could 0 pad, but for now instead just ensure kernelsize and offset are such that
            #thre are no encoder-side issues with kernel extending beyond history (relevant for very short histories)
            #Offset can stay the same (represents number of positions to RIGHT of given position),
            #but adjust kernel size to fit:
            self.offset = hparams.MLP_POSTPROCESS__KERNEL_OFFSET
            leftside = hparams.MLP_POSTPROCESS__KERNEL_SIZE -1 -self.offset
            self.actual_kernel_size = min(leftside, hparams.history_window_size_minmax[0]) +1 +self.offset
            decoder_targets = self.MLP_postprocess(decoder_targets, inp.time_y, 
                                                   inp.time_x, 
                                                   summary_z if hparams.RECURSIVE_W_ENCODER_CONTEXT else None, 
                                                   self.actual_kernel_size, self.offset, None)
#        decoder_targets = tf.Print(decoder_targets,['decoder_targets AFTER postprocess',decoder_targets,'decoder_outputs',decoder_outputs])        
#        decoder_targets = tf.Print(decoder_targets,['encoder_state',encoder_state,'inp.time_y',inp.time_y,'inp.norm_x',inp.norm_x])
#        decoder_targets = tf.Print(decoder_targets,['decoder_targets',decoder_targets,'decoder_outputs',decoder_outputs])
        
        #Vs. if doing the direct MLP decoder (while loop of MLP's instead of while loop of RNN cells):
        elif self.MLP_DIRECT_DECODER:
            #LOCAL_CONTEXT_SIZE=64,
            #GLOBAL_CONTEXT_SIZE=256,
            pass
        
        
        
        # Decoder activation losses
        dec_stab_loss = rnn_stability_loss(decoder_outputs, hparams.decoder_stability_loss / inp.horizon_window_size)
        dec_activation_loss = rnn_activation_loss(decoder_outputs, hparams.decoder_activation_loss / inp.horizon_window_size)
        print('dec_stab_loss',dec_stab_loss)
        print('dec_activation_loss',dec_activation_loss)

        # Get final denormalized predictions
        self.predictions = decode_predictions(decoder_targets, inp)
#        vv = decode_predictions(decoder_targets, inp)
#        vv = tf.Print(vv, ['decode_predictions',vv,tf.shape(vv)])
#        self.predictions = vv
#        print('self.predictions (still log1p(counts))')
        print(self.predictions)
        
        
        #!!!!!! for now just see if helps. ignore last nodes, only use 0th.
        self.predictions = self.predictions[0] #Now is batch x time
        

        # Calculate losses and build training op
        if inp.mode == ModelMode.PREDICT:
            # Pseudo-apply ema to get variable names later in ema.variables_to_restore()
            # This is copypaste from make_train_op()
            if asgd_decay:
                self.ema = tf.train.ExponentialMovingAverage(decay=asgd_decay)
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if graph_prefix:
                    ema_vars = [var for var in variables if var.name.startswith(graph_prefix)]
                else:
                    ema_vars = variables
                self.ema.apply(ema_vars)
        else:
            self.mae, smape_loss, self.smape, self.loss_item_count, self.ave_quantile_loss, self.quantile_losses = calc_loss(self.predictions, inp.true_y,
                                                                               additional_mask=loss_mask,
                                                                               DO_QUANTILES=self.hparams.DO_QUANTILES,
                                                                               QUANTILES=self.hparams.QUANTILES)
            #from calc_loss:
            #mae_loss, smape_loss(true_y, predictions, weights), calc_smape_rounded(true_y, predictions, weights), tf.size(true_y)
            
            
            if is_train:
                # Sum all losses
                total_loss = smape_loss + enc_stab_loss + dec_stab_loss + enc_activation_loss + dec_activation_loss
                #For quantile regression: just include a term for sum over all quantile losses [weights every quantile equally]
                if self.hparams.DO_QUANTILES:
                    total_loss += self.LAMBDA * self.ave_quantile_loss
                self.train_op, self.glob_norm, self.ema = make_train_op(total_loss, asgd_decay, prefix=graph_prefix)



    def default_init(self, seed_add=0):
        return default_init(self.seed + seed_add)

    def decoder(self, encoder_state, attn_features, summary_z, prediction_inputs, previous_y):
        """
        :param encoder_state: shape [batch_size, encoder_rnn_depth]
        :param prediction_inputs: features for prediction days, tensor[batch_size, time, input_depth]
        :param previous_y: Last day pageviews, shape [batch_size, self.lookback_K_actual] 
        :param attn_features: Additional features from attention layer, shape [batch, horizon_window_size, readout_depth*n_heads]
        :return: decoder rnn output
        """
        hparams = self.hparams

        def build_cell(idx):
            with tf.variable_scope('decoder_cell', initializer=self.default_init(idx)):
                print('self.hparams.rnn_depth',self.hparams.rnn_depth)
                
                cell = rnn.GRUBlockCell(self.hparams.rnn_depth) #GRUBlockCellV2 same butnewer version just variables named differently?
                has_dropout = hparams.decoder_input_dropout[idx] < 1 \
                              or hparams.decoder_state_dropout[idx] < 1 or hparams.decoder_output_dropout[idx] < 1

                #context size alone may be as big as decoder state size?? Then input-> hidden would be a down projection...
                #so maybe do a projection down, on the encoder side first [e.g. encoder output??] then better here...
                if self.is_train and has_dropout:
                    attn_depth = attn_features.shape[-1].value if attn_features is not None else 0
                    context_depth = summary_z.shape[-1].value if self.hparams.RECURSIVE_W_ENCODER_CONTEXT else 0 #Should just be the encoder RNN depth
                    print('attn_depth',attn_depth, 'context_depth',context_depth)
                    input_size = attn_depth + context_depth + prediction_inputs.shape[-1].value + self.lookback_K_actual if idx == 0 else self.hparams.rnn_depth
                    input_size = tf.Print(input_size, ['attn_depth',tf.shape(attn_depth),attn_depth, 'context_depth',tf.shape(context_depth),context_depth, 'input_size',tf.shape(input_size),input_size])#!!!!!!!!!!
                    cell = rnn.DropoutWrapper(cell, dtype=tf.float32, input_size=input_size,
                                              variational_recurrent=hparams.decoder_variational_dropout[idx],
                                              input_keep_prob=hparams.decoder_input_dropout[idx],
                                              output_keep_prob=hparams.decoder_output_dropout[idx],
                                              state_keep_prob=hparams.decoder_state_dropout[idx], seed=self.seed + idx)
                return cell

        if hparams.decoder_rnn_layers > 1:
            cells = [build_cell(idx) for idx in range(hparams.decoder_rnn_layers)]
            cell = rnn.MultiRNNCell(cells)
        else:
            cell = build_cell(0)


        #!!!!!! on our data, when doing side_split, encoder_state is fine [no NANs],
        #but when doing walk_forward, some rows (instances) are all NANs (and the others all defined),
        #then eventually every instance becomes NANs
#        N_nans = tf.reduce_sum(tf.cast(tf.is_nan(encoder_state), tf.float32))
#        tt = tf.cast(tf.is_nan(encoder_state), tf.float32)
#        ff = tf.reduce_sum(tt,axis=1)
#        ggg = tf.cast(tf.equal(ff, ff*0.+267.), tf.float32)
#        N_all_NAN_encoder_states = tf.reduce_sum(ggg)
#        total = tf.reduce_prod(tf.shape(encoder_state))
#        encoder_state = tf.Print(encoder_state,['encoder_state', tf.shape(encoder_state), encoder_state, 'N_nans', N_nans, 'total', total, 'N_all_NAN_encoder_states', N_all_NAN_encoder_states])
        


        nest.assert_same_structure(encoder_state, cell.state_size)
#        predict_timesteps = 7777#tf.reshape(self.inp.horizon_window_size, [])
        predict_timesteps = self.inp.horizon_window_size
#        print('predict_timesteps',predict_timesteps,type(predict_timesteps))
        #When random size steps, cannot know dims in advance to do below assert:
        assert prediction_inputs.shape[1] == predict_timesteps #!!!!!!!quantiles
#        print('predict_timesteps',predict_timesteps,'prediction_inputs.shape[1]',prediction_inputs.shape)

        # [batch_size, time, input_depth] -> [time, batch_size, input_depth]
        inputs_by_time = tf.transpose(prediction_inputs, [1, 0, 2])

        # Return raw outputs for RNN losses calculation
        return_raw_outputs = self.hparams.decoder_stability_loss > 0.0 or self.hparams.decoder_activation_loss > 0.0

        # Stop condition for decoding loop
        def cond_fn(timestep, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            return timestep < predict_timesteps #If doing k2-step lookahead prediction for k2>1, possibly want to 
            #adjust condition to do appropriate n steps > predict_timesteps... and then combine predictions for those steps to get single prediction, 
            #e.g. by exponential weighting  backward in time from this step.

        # FC projecting layer to get single predicted value from RNN output
        def project_output(tensor):
            N_pctls=1 #!!!!!!!!!! quantiles
            return tf.layers.dense(tensor, N_pctls, name='decoder_output_proj', kernel_initializer=self.default_init())

        def loop_fn(timestep, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            """
            Main decoder loop
            :param timestep: timestep number
            :param prev_output: Output(prediction) from previous step --> from previous K steps: self.lookback_K_actual 
            :param prev_state: RNN state tensor from previous step
            :param array_targets: Predictions, each step will append new value to this array
            :param array_outputs: Raw RNN outputs (for regularization losses)
            :return:
            """
            # RNN inputs for current step
            features = inputs_by_time[timestep]
#            print('features',features)
#            print('previous_y',previous_y)

            # [batch, horizon_window_size, readout_depth * n_heads] -> [batch, readout_depth * n_heads]
            if attn_features is not None:
                #  [batch_size, 1] + [batch_size, input_depth]
                attn = attn_features[:, timestep, :]
                # Append previous predicted value + attention vector to input features
                next_input = tf.concat([prev_output, features, attn], axis=1)
               
            else:
                # Append previous predicted value to input features
                next_input = tf.concat([prev_output, features], axis=1)
            #If using more of a typical encoder-decoder, also have encoder context each time:
            if self.hparams.RECURSIVE_W_ENCODER_CONTEXT:
                next_input = tf.concat([next_input, summary_z], axis=1) #!!!!!!!!summary_z[-1]
                    
            print('next_input',next_input)
            print('prev_state',prev_state)            
            # Run RNN cell
            output, state = cell(next_input, prev_state)
            # Make prediction from RNN outputs
            projected_output = project_output(output) #!!!!!!!!!! quantiles
#            projected_output = tf.Print(projected_output, ['timestep',timestep,'projected_output',projected_output,tf.shape(projected_output),'output',output,tf.shape(output),'state',state,tf.shape(state) ,'prev_output',prev_output,tf.shape(prev_output) ,'features',features,tf.shape(features),features[1,:18]])
            
            # Append step results to the buffer arrays
            if return_raw_outputs:
                array_outputs = array_outputs.write(timestep, output)
            array_targets = array_targets.write(timestep, projected_output)
            
            #Update prev_output
            #(delete oldest left, append rightmost)
            if self.lookback_K_actual > 1:
                prev_output = prev_output[:,1:] #All examples in batch, exclude oldest output [leftmost oldest, rightmost most recent]
#                print('prev_output',prev_output)
#                print('projected_output',projected_output)
                updated_outputs = tf.concat([prev_output,projected_output],axis=1)
#                print('updated_outputs',updated_outputs)
            elif self.lookback_K_actual==1:
                updated_outputs = prev_output
                
            # Increment timestep and return
            return timestep + 1, updated_outputs, state, array_targets, array_outputs #!!!!!! quantiles: projected_output will be diff dims

        # Initial values for loop
        loop_init = [tf.constant(0, dtype=tf.int32), #timestep
#                     previous_y if self.lookback_K_actual  > 1 else tf.expand_dims(previous_y, -1), #prev_output
                    previous_y, #prev_output
                     encoder_state, #prev_state
                     tf.TensorArray(dtype=tf.float32, size=predict_timesteps), #array_targets
                     tf.TensorArray(dtype=tf.float32, size=predict_timesteps) if return_raw_outputs else tf.constant(0)] #array_outputs #!!!!!!! size= ... x N_pctls
        # Run the loop
        _timestep, _projected_output, _state, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn, loop_init)
        
        #!!!!!!!!if do the decoder as fixed max predict window, then here, when in train mode,  slice to get only those really evaluated
        
        
        
        print('decoder')
#        print('_timestep',_timestep)
#        _timestep = debug_tensor_print(_timestep)
#        print('_projected_output',_projected_output)
#        _projected_output = debug_tensor_print(_projected_output)     
#        print('_state',_state)        
#        _state = debug_tensor_print(_state)  

        
#        targets_ta_tensor = tf.convert_to_tensor(targets_ta)
#        targets_ta_tensor = tf.Print(targets_ta_tensor,[targets_ta_tensor])
#        print('targets_ta',targets_ta)
#        print('outputs_ta',outputs_ta)
        
        
        # Get final tensors from buffer arrays
#        targets_ta = tf.Print(targets_ta,['targets_ta',targets_ta,tf.shape(targets_ta)])
#        print('targets_ta',targets_ta)
        targets = targets_ta.stack()
        # [time, batch_size, 1] -> [time, batch_size]
        targets = tf.squeeze(targets, axis=-1)
#        print('targets after squeeze',targets)
#        targets = tf.Print(targets,['targets after squeeze',targets,tf.shape(targets)])
        raw_outputs = outputs_ta.stack() if return_raw_outputs else None

#        print('targets',targets)
        #!!!!!!!!!!! why targets becomes NANs ?????
#        why targets NANs?
#        targets = debug_tensor_print(targets)  #63 x 245,   except for first 2 prints for each new iteration it is 63 x 64
#        raw_outputs = debug_tensor_print(raw_outputs) #is 63 x 64 x 267
        
#        print_list = ['_timestep', _timestep.name, tf.shape(_timestep), _timestep]
#        raw_outputs = tf.Print(raw_outputs, print_list)
        
        return targets, raw_outputs
    
    
    
    def MLP_postprocess(self, decoder_targets, decoder_features, time_x, summary_z, cropped_kernel_size, offset, seed):#kernel_size, kernel_offset, seed):
        """
        As a postprocess decoder step:
        
        After the decoder has output predictions for each decoder timestep 
        (in standard mode this is a vector of predictions: one per timestep
        vs. if doing quantile regression, is a matrix timesteps x quantiles...)
        
        Refine those predictions by taking into account the neighboring predictions.
        Ideally, the RNN/LSTM/GRU state would contain all necessary information about state 
        to make good predictions. In reality, might be asking too much, could be useful 
        to do a second pass over the outputs to improve predictions.
        
        E.g. if encoding shock events: binary 0,1 with 1 on change day:
        if there is no pre-event shoulder, no information on Monday takes advantage 
        of known change about to ocur on Tuesday. This postprocessor will help with that.
        
        
        hparams.QUANTILES = [None],#[.20, .30, .40, .50, ,75, .90]
        hparams.MLP_POSTPROCESS__KERNEL_SIZE=15,
        hparams.MLP_POSTPROCESS__KERNEL_OFFSET=7,
        
        
        
        INPUTS:
            decoder_targets - [HORIZON x BATCH]  (for now ignoring quantiles just single value per decoder step)
            decoder_features - decoder side features for all horizon timesteps [BATCH x HISTORY x FEATURES] (but gets transposed later to make easier)
            time_x - encoder side features and ground truth for all history timesteps [BATCH x HISTORY x FEATURES+1] (but gets transposed later to make easier)
            summary_z - encoded context vector, simple way using same vector as context for every decoder step [BATCH x RNNSTATEDEPTH]. Will be None if not doing encoder context option
            
        
            kernel_size - int
            The size, in #timesteps, of the MLP postprocess kernel.
            For a given timestep of decoder, a window containing that day is used to 
            hard select which nearby timesteps are allowed to contribute to refinement 
            for that timestep. I.e. bigger kernel means looking at more nearby RNN outputs.
    
            kernel_offset - int
            The offset, in #timesteps, of the MLP postprocess kernel from the given timestep under consideration.
            0 offset means kernel has rightmost index exactly on the given timestep,
            1 offset means the rightmost timestep included in the postprocess window is 
            the timestep exactly 1 position right of the timestep under consideration, ..., 
            ...
            K-1 is when the kernel window is trasnlated as far as possible to the right,
            i.e. has the given timestep as the LEFTmost position, and K-1 positions to the right of the day in question.
            In other words, offset tells you how many positions to the right of the given timestep are included in the kernel window.
        
        
        For this mini feedforward net, just use 1 hidden layer with nonlinearity, and for #of nodes there, just make it half of #input nodes to this MLP.
                
        ALternative: you could also choose to simplify things a bit on the left side (early in time)
        where there is overlap with encoder side and even potentially with before the encoder starts (which needs padding).
        To simplify, could choose to not do refinement on those earliest outputs, and only do refinement when kernel
        is fully in decoder area. Justification could be that refinement of those early predictions is less needed
        since they are closer to the forecast creation time, so should suffer less from the recursion problem and hopefully be OK already.
        """
        
        
        #Get dimensions of things based on inputs, features, options:
        #N1 is number nuerons in input layer, is   (KERNELSIZE x FEATURES) + KERNELSIZE (+ENCODER_DEPTH if encoder_context)
        context_depth = summary_z.shape[-1].value if self.hparams.RECURSIVE_W_ENCODER_CONTEXT else 0 #Should just be the encoder RNN depth
        feature_depth = decoder_features.shape[-1].value
        batchsize = tf.shape(decoder_features)[0]
        N1 =  cropped_kernel_size*(feature_depth+1) + context_depth + 2 #+2 is to also use two timestep index features, this is since at beginning/end of iterations, could have overlap with 0-padded region, and those nodes of net would normally have a very different distribution of nonzero values, so signal this.
        N2 = int(min(max(.5*N1,100),400)) #Use between 100-400 neurons in layer 2, using approximately 
        print('MLP Post-processing N1, N2:', N1, N2)
        #0-pad the right side (for now just make offset/kernel size such that don't need to worry about left side before encoder)
        #The amount of padding needed is based on horizon, kernel size, offset:
        print('offset',offset)
        print('batchsize',batchsize)
        #_ = tf.stack([offset, tf.shape(decoder_features)[0], feature_depth])
        _ = tf.stack([offset, batchsize, feature_depth+1])
        right_pad_zeros = tf.fill(_, 0.0)
        
        

        
        #If doing quantile regression
        Nquantiles = len(self.hparams.QUANTILES) if self.hparams.DO_QUANTILES else 0
        N_out = 1 + Nquantiles #+1 because also do a point estimate optimized by SMAPE. (Even if some of the wuantiles e.g. .48 are really for point estimates, still keep the direct SMAPE optimized point estimate as well.)
        print('Nquantiles',Nquantiles)
        print('N_out',N_out)
        

        #Quick check there are correct number timesteps:
        assert decoder_features.shape[1] == self.inp.horizon_window_size #!!!!!!!quantiles

        #Rearrange tensors to all have time first:
        # [batch_size, time, input_depth] -> [time, batch_size, input_depth]
        decoder_targets = tf.expand_dims(decoder_targets,axis=-1)
        decoder_features = tf.transpose(decoder_features, [1, 0, 2])
        print('decoder_targets',decoder_targets)
        print('decoder_features',decoder_features)        
        
        decoder_by_time = tf.concat([decoder_targets,decoder_features],axis=2)
        encoder_by_time = tf.transpose(time_x, [1, 0, 2]) #Has target and features stacked already

        print('decoder_by_time',decoder_by_time)
        print('encoder_by_time',encoder_by_time)
        all_features_targets_by_time = tf.concat([encoder_by_time,decoder_by_time,right_pad_zeros],axis=0)
        print('all_features_targets_by_time',all_features_targets_by_time)
        
        
        #Simple 2 layer feedforward
        def feedforward(_x):
            fc1 = tf.layers.dense(_x, N2, activation=selu, name='fc1', kernel_initializer=self.default_init())
            fc2 = tf.layers.dense(fc1, N_out, name='fc2', kernel_initializer=self.default_init())
            return fc2
        
        
        
#        #Simple 2 layer feedforward
#        def feedforward(kernel_window_cropped):
##            _x = tf.placeholder(tf.float32, [None, N1])
#            def network():
#                with tf.Graph().as_default():
#                    _x = tf.placeholder(tf.float32, [None, N1])
#                    with tf.variable_scope("MLP_postprocess", 
#                                       initializer=layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', seed=seed)):
#                        
##                        tf.layers.Input(shape=(N1,), dtype=tf.float32)
#                        print('_x',_x)
#                        fc1 = tf.layers.dense(_x, N2, activation=selu, name='fc1')
#                        print('fc1',fc1)
#                        fc2 = tf.layers.dense(fc1, N_out, name='fc2')
#                        print('fc2',fc2)
#    #                    out = tf.layers.Network(kernel_window_cropped, fc2)
#                        return fc2
#                    
#            with tf.Graph().as_default():
#                net = network()
##                config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
##                with tf.Session(config=config) as sess:
#                with tf.Session() as sess:                
#                    result = sess.run(net,feed_dict={_x: kernel_window_cropped})
#                    return result
                
        
        
        
        
        
        
        
        
        
        
        
        
    
    
        # Stop condition for decoding loop
        def cond_fn(timestep, all_timesteps_input, array_targets: tf.TensorArray):
            return timestep < self.inp.horizon_window_size

        def loop_fn(timestep, all_timesteps_input, array_targets: tf.TensorArray):
            print(timestep)
            #Excerpt time window
            start = timestep - cropped_kernel_size + 1 + offset + self.inp.history_window_size
            end = timestep + offset + self.inp.history_window_size + 1
            cropped = all_features_targets_by_time[start:end,:,:]
            print('timestep,start,end', timestep,start,end)
            cropped = tf.transpose(cropped,[1,0,2])
            cropped = tf.contrib.layers.flatten(cropped)
            
            #If using context vector, also append it
            if self.hparams.RECURSIVE_W_ENCODER_CONTEXT:
                cropped = tf.concat([cropped,summary_z],axis=1)

            print('cropped',cropped)
            
            #Include timestep related features:
            #Because using fixed dimension MLP with fixed input for input layer neurons,
            #but window is sliding over boundaries with different meaning (0 padded vs legit values),
            #use 2 additional features that help control issue of boundary effects.
            #Use the (normalized) percent through decoder phase, = normalize(timestep/horizon),
            #and the (normalized) percent overlap of the kernel with the 0padded region
            f1 = timestep/self.inp.horizon_window_size - .5
            f2 = tf.maximum(0, offset - (self.inp.horizon_window_size - timestep)) / offset - .5
            _ = tf.stack([batchsize,1])
            f1 = tf.cast(tf.fill(_, f1),tf.float32)
            f2 = tf.cast(tf.fill(_, f2),tf.float32)
            print('f1',f1)
            print('f2',f2)
            print('cropped',cropped)
            flattened_input = tf.concat([cropped,f1,f2],axis=-1)
            #Before even running this, we know the shape (other than batchsize which may be dynamic)
            #So, since dense layer in feed_forward needs last dimension specified, do this:
            flattened_input.set_shape((None,N1))
            print('flattened_input',flattened_input)

            #Pass tensor into the simple feedforward network:
            projected_output = feedforward(flattened_input)
            #Append this timestep results 
            array_targets = array_targets.write(timestep, projected_output)
            
                
            return timestep + 1, all_timesteps_input, array_targets #!!!!!! quantiles: projected_output will be diff dims


        #Since we know the dimensions beforehand, jsut use an init with defined dimension so can initialize the feedforward net
#        _ = tf.stack([batchsize,N1])
#        init_features_zeros = tf.cast(tf.fill(_, 0.),tf.float32)
#        print('init_features_zeros',init_features_zeros)
        
        # Initial values for loop
        loop_init = [tf.constant(0, dtype=tf.int32), #timestep
                    all_features_targets_by_time, #init_features_zeros,#all_features_targets_by_time, #all_timesteps_input
                    tf.TensorArray(dtype=tf.float32, size=self.inp.horizon_window_size)] #array_targets

        # Run the loop
        _timestep, _, targets_ta = tf.while_loop(cond_fn, loop_fn, loop_init)        
        
        #Get the post-processed predictions
        #[time, batch_size, Nptcl]
        targets = targets_ta.stack()
        
        # [time, batch_size, 1] -> [time, batch_size]
#        targets = tf.squeeze(targets, axis=-1)
                
        return targets    





#Direct MLP decoder...
#            MLP_DIRECT_DECODER=False,
#    LOCAL_CONTEXT_SIZE=64,
#    GLOBAL_CONTEXT_SIZE=256,