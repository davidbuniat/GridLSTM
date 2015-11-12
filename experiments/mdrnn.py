#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''


#THEANO_FLAGS=device=gpu python -c "import theano; print theano.sandbox.cuda.device_properties(0)"

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


import png
import os
import random

import itertools


DEBUG = False


def load_dataset():
    number_training = 455
    row_count = 120
    column_count = 160
    plane_count = 3
    print ("Loading data...")
    # We first define a download function, supporting both Python 2 and 3.
#    if sys.version_info[0] == 2:
#        from urllib import urlretrieve
#    else:
#        from urllib.request import urlretrieve
#
#    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
#        print("Downloading %s" % filename)
#        urlretrieve(source + filename, filename)
#
    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.

    def load_images(pathname):
        
        cur_dir = os.path.dirname(os.path.realpath(__file__)) #Current Directory
        
        listing = os.listdir(cur_dir+pathname)
        p = [(255,0,0, 0,255,0, 0,0,255),
            (128,0,0, 0,128,0, 0,0,128)]  

        data = np.empty((number_training,row_count*column_count, plane_count))
        index = 0
        for file in listing:
            r = png.Reader(cur_dir+pathname+'/'+file)      # binary mode is important
            pngData = r.read()[2]

            image_2d = np.vstack(itertools.imap(np.uint16, pngData))
            image_3d = np.reshape(image_2d, (row_count,column_count, plane_count))

            image_1d = np.reshape(image_2d, (row_count*column_count, plane_count))
            data[index] = image_1d
            index = index + 1
            #For debuggin
            if DEBUG and index>9:
                break
        
        return data[:index]

    # We read the training and test set images and labels.
    X = load_images('/afreightdata/data')
    Y = load_images('/afreightdata/label')
   
    def labelling(y):
        labels = {}
        index = 0
        new_y = np.empty((number_training, row_count*column_count, 1))
        
        for i in xrange(1,len(y)):
            for j in xrange(1,len(y[i])):
                id = str(y[i][j][0])+str(y[i][j][1])+str(y[i][j][2])

                if id in labels:
                    new_y[i][j][0] = labels[id]
                else:
                    labels[id]= index
                    index = index + 1
                    new_y[i][j][0] = labels[id]
                pass
            pass

        print("labels added " +str(index))
        new_y = new_y/np.float32(str(index))
          
        return new_y

    # Normalizing Data
    X = X / np.float32(256)
    Y = labelling(Y)
    print()


    train_count = 250
    val_count = 150
    test_count = 55

    if DEBUG:
        train_count = 5
        val_count = 3
        test_count = 2

    rand_array = random.sample(xrange(0,len(X)), len(X));
    val_array = rand_array[:val_count]
    test_array = rand_array[:test_count]
    train_array = rand_array[:train_count]

    print("Loaded "+str(len(X)) + " images")
    return X[train_array], Y[train_array], X[val_array], Y[val_array], X[test_array], Y[test_array]


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 19200
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 4
# Number of training sequences in each batch
N_BATCH = 250
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 25

if DEBUG:
    NUM_EPOCHS = 1
    EPOCH_SIZE = 5

BATCH_SIZE = 455
length = 19200




def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer((N_BATCH, MAX_LENGTH, 3))
    batchsize, seqlen, _ = l_in.input_var.shape
    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    l_forward = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_backward = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh, backwards=True)


    l_recurrent = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])

    l_reshape = lasagne.layers.ReshapeLayer(l_recurrent,
                                       (batchsize*seqlen, N_HIDDEN))

    #nonlinearity = lasagne.nonlinearities.softmax

    l_rec_out = lasagne.layers.DenseLayer(l_reshape, num_units=1)

    l_out = lasagne.layers.ReshapeLayer(l_rec_out,
                                    (batchsize, seqlen, 1))

    # Now, we'll concatenate the outputs to combine them.
    #l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward], 2)

    #l_shp = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))

    # Our output layer is a simple dense connection, with 1 output unit
    #l_final = lasagne.layers.DenseLayer(l_shp, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
    
    #l_out = lasagne.layers.ReshapeLayer(l_final, (batchsize, seqlen, 1))

    target_values = T.tensor3('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The value we care about is the final value produced for each sequence
    predicted_values = network_output

    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
    
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values], cost)

    get_prediction = theano.function([l_in.input_var], predicted_values)

    # We'll use this "validation set" to periodically check progress
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
 
    print("Training ...")
    #print(get_prediction(X_train[0:1]))

    try:
        index = 0
        for epoch in range(num_epochs):
            X = X_train[EPOCH_SIZE*epoch:EPOCH_SIZE*(epoch+1)]
            y = y_train[EPOCH_SIZE*epoch:EPOCH_SIZE*(epoch+1)]
            train(X, y)

            cost_val = compute_cost(X_val, y_val)
            cost_test = compute_cost(X_test, y_test)
            print("Epoch {} validation cost = {}  test cost = {} ".format(epoch, cost_val, cost_test))
    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
