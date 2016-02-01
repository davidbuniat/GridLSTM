#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Loading Data
from __future__ import print_function

import scipy
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import png
import os
import random
import itertools
import lasagne

labels = {}
labels_rev = {}
index_temp = 0
mean = 0
std = 0  
def one_hot(x, m):
    y = [0] * m
    y[x] = 1
    return y


def load_dataset(number_training, row_count, column_count, plane_count, DEBUG):
    print ("Loading data...")
    # We first define a download function, supporting both Python 2 and 3.

    # We then define functions for loading afreightData images and labels.
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
    X = load_images('/../datasets/afreightdata/data')
    Y = load_images('/../datasets/afreightdata/label')
    def one_hot(x, m):
        y = np.zeros(m)
        y[x] = 1
        return y

    def labelling(y):
        new_y = np.empty((number_training, row_count*column_count,157))
        index = 0
        for i in xrange(1,len(y)):
            for j in xrange(1,len(y[i])):
                id = str(y[i][j][0])+'-'+str(y[i][j][1])+"-"+str(y[i][j][2])

                if id in labels:
                    #new_y[i][j] = one_hot(labels[id],157)
                    if(labels[id]<60):
                        new_y[i][j][0] = 0
                    else:
                        new_y[i][j][0] = 1

                else:
                    labels[id] = index
                    labels_rev[index] = y[i][j]
                    index = index + 1
                    new_y[i][j] = one_hot(labels[id],157)
                    if(labels[id]<60):
                        new_y[i][j][0] = 0
                    else:
                        new_y[i][j][0] = 1
                pass
            pass

        print("labels added " +str(index))
        #new_y = new_y/np.float32(index)
        #clearnew_y = lasagne.utils.one_hot(new_y)
        #print(new_y)
        index_temp = index
          
        return new_y

    # Normalizing Data
    X = (X / np.float32(256))*2 - 1
    Y = labelling(Y)

    train_count = 250
    val_count = 150
    test_count = 55

    if DEBUG:
        train_count = 5
        val_count = 3
        test_count = 2

    rand_array = random.sample(xrange(0,len(X)), len(X));
    val_array = np.array(rand_array[:val_count])
    test_array = np.array(rand_array[:test_count])
    train_array = np.array(rand_array[:train_count])

    print("Loaded "+str(len(X)) + " images")
    return X[train_array], Y[train_array], X[val_array], Y[val_array], X[test_array], Y[test_array]

def saveImage(Y, name, row_count, column_count, plane_count, predicted=False):
    new_y = np.empty((row_count*column_count, plane_count))
    for i in range(len(new_y)):
        if predicted:

            new_y[i] = labels_rev[np.argmax(Y[i])]
        else:
            new_y[i] = labels_rev[int(round(Y[i][0]))]
    image = np.reshape(new_y, (row_count, column_count, plane_count))
    plt.imsave(name, image)


