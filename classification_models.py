# -*- coding: utf-8 -*-
"""
Dilated LSTM Cells in Tensorflow

From github user "code-terminator"
https://github.com/code-terminator/DilatedRNN/blob/master/models/drnn.py

based on the paper
Dilated Recurrent Neural Networks
Nov 2017
Chang et al.
https://arxiv.org/pdf/1710.02224.pdf 
"""

import tensorflow as tf
from drnn import multi_dRNN_with_dilations

def _contruct_cells(hidden_structs, cell_type):
    """
    This function contructs a list of cells.
    """
    # error checking
    if cell_type not in ["RNN", "LSTM", "GRU"]:
        raise ValueError("The cell type is not currently supported.")

    # define cells
    cells = []
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(hidden_dims)
        elif cell_type == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_dims)
        elif cell_type == "GRU":
            cell = tf.contrib.rnn.GRUCell(hidden_dims)
        cells.append(cell)

    return cells


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)

    return x_reformat


def drnn_classification(x,
                        hidden_structs,
                        dilations,
                        n_steps,
                        n_classes,
                        input_dims=1,
                        cell_type="RNN"):
    """
    This function construct a multilayer dilated RNN for classifiction.  
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        n_steps -- the length of the sequence.
        n_classes -- the number of classes for the classification.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].
    
    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions.
    """
    # error checking
    assert (len(hidden_structs) == len(dilations))

    # reshape inputs
    x_reformat = _rnn_reformat(x, input_dims, n_steps)

    # construct a list of cells
    cells = _contruct_cells(hidden_structs, cell_type)

    # define dRNN structures
    layer_outputs = multi_dRNN_with_dilations(cells, x_reformat, dilations)

    if dilations[0] == 1:
        # dilation starts at 1, no data dependency lost
        # define the output layer
        weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1],
                                                      n_classes]))
        bias = tf.Variable(tf.random_normal(shape=[n_classes]))
        # define prediction
        pred = tf.add(tf.matmul(layer_outputs[-1], weights), bias)
    else:
        # dilation starts not at 1, needs to fuse the output
        
        # define output layer
        weights = tf.Variable(tf.random_normal(shape=[hidden_structs[
            -1] * dilations[0], n_classes]))
        bias = tf.Variable(tf.random_normal(shape=[n_classes]))

        # concat hidden_outputs
        for idx, i in enumerate(range(-dilations[0], 0, 1)):
            if idx == 0:
                hidden_outputs_ = layer_outputs[i]
            else:
                hidden_outputs_ = tf.concat(
                    [hidden_outputs_, layer_outputs[i]],
                    axis=1)

        pred = tf.add(tf.matmul(hidden_outputs_, weights), bias)

    return pred