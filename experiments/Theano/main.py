
from __future__ import print_function

import scipy
import numpy as np
import theano
import theano.tensor as T
import lasagne

import png
import os
import random
import itertools

import data as dt

#Parameters
DEBUG = False
number_training = 455
row_count = 120
column_count = 160
plane_count = 3

# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 19200
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 8
# Number of training sequences in each batch
N_BATCH = 10
# Optimization learning rate
LEARNING_RATE = 0.000001
MOMENTUM  = 0.9

# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 15

if DEBUG:
    NUM_EPOCHS = 1
    EPOCH_SIZE = 5

BATCH_SIZE = 455
length = 19200


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")


    l_in = lasagne.layers.InputLayer((None, MAX_LENGTH, 3))
    batchsize, seqlen, _ = l_in.input_var.shape

    l_forward = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_backward = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh, backwards=True)

    l_recurrent = lasagne.layers.ElemwiseMergeLayer([l_forward, l_backward], T.mul)

    softmax = lasagne.nonlinearities.softmax

    l_reshape = lasagne.layers.ReshapeLayer(l_recurrent,(-1, N_HIDDEN))

    l_drop_out = lasagne.layers.DropoutLayer(l_reshape, p=0.95)

    l_dense = lasagne.layers.DenseLayer(l_drop_out, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    l_drop_out_2 = lasagne.layers.DropoutLayer(l_dense, p=0.95)

    #l_drop_out_2 = lasagne.layers.DropoutLayer(l_reshape, p=0.5)

    l_softmax = lasagne.layers.DenseLayer(l_drop_out_2, num_units=2, nonlinearity = softmax)

    l_out = lasagne.layers.ReshapeLayer(l_softmax, (batchsize, seqlen, 2)) 

    # Now, we'll concatenate the outputs to combine them.
    #l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward], 2)

    #l_shp = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))

    # Our output layer is a simple dense connection, with 1 output unit
    #l_final = lasagne.layers.DenseLayer(l_shp, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
    
    #l_out = lasagne.layers.ReshapeLayer(l_final, (batchsize, seqlen, 1))

    target_values = T.tensor3('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The value we care about is the final value produced  for each sequence
    #predicted_values = T.argmax(network_output, axis = 2, keepdims = True)
    predicted_values = network_output

    # Our cost will be mean-squared error
    cost = T.mean((T.argmax(predicted_values, axis = 2, keepdims = True) - target_values)**2)
    #cost = lasagne.objectives.squared_error(T.argmax(predicted_values, axis = 2)+1, target_values).mean()
    #cost = cost.mean()

    acc = T.mean(T.eq(T.argmax(predicted_values, axis = 2, keepdims = True), target_values),
                      dtype=theano.config.floatX)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.nesterov_momentum(cost, all_params, learning_rate=LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values], cost)
    compute_acc = theano.function(
        [l_in.input_var, target_values], acc)

    get_out = T.argmax(predicted_values, axis = 2, keepdims = True)
    get_prediction = theano.function([l_in.input_var], get_out)
    get_prediction_2 = theano.function([l_in.input_var], predicted_values)

    # We'll use this "validation set" to periodically check progress
    X_train, y_train, X_val, y_val, X_test, y_test = dt.load_dataset(BATCH_SIZE, row_count, column_count, plane_count, DEBUG)
 
    print("Training ...")
    #print(get_prediction(X_train[0:1]))

    try:
        index = 0 #*len(dt.labels_rev)
        dt.saveImage(y_test[0], "results/y_GT.png",row_count, column_count,  plane_count)
        for epoch in range(num_epochs):
            X = X_train[EPOCH_SIZE*epoch:EPOCH_SIZE*(epoch+1)]
            y = y_train[EPOCH_SIZE*epoch:EPOCH_SIZE*(epoch+1)]
            train(X, y)

            cost_val = compute_cost(X_val, y_val)
            cost_test = compute_acc(X_test, y_test)*100
            #print(y_test[0]) 
            #print(get_prediction(X_test)[0])
            #print(get_prediction_2(X_test)[0]) 
            print("Epoch {} validation cost = {}  test acc = {} %".format(epoch, cost_val, cost_test))

            dt.saveImage(get_prediction(X_test)[0], "results/y_output_{}.png".format(epoch), row_count, column_count,  plane_count, True)

        dt.saveImage(get_prediction(X_test)[0], "results/y_output.png", row_count, column_count,  plane_count, True)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
