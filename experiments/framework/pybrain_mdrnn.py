from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.


from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SupervisedDataSet,SequenceClassificationDataSet
from pybrain.structure.modules   import MDLSTMLayer, LSTMLayer, SoftmaxLayer, LinearLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.networks  import RecurrentNetwork
from pybrain.structure           import FullConnection

#from .datasets import generateNoisySines

import numpy as np
from numpy.random import multivariate_normal, rand
from scipy import diag
from pylab import show, hold, plot




import png
import os
import random

import itertools

DEBUG = True;


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

        data = np.empty((number_training,row_count,column_count, plane_count))
        index = 0
        for file in listing:
            r = png.Reader(cur_dir+pathname+'/'+file)      # binary mode is important
            pngData = r.read()[2]

            image_2d = np.vstack(itertools.imap(np.uint16, pngData))
            image_3d = np.reshape(image_2d, (row_count, column_count, plane_count))

            #image_1d = np.reshape(image_2d, (row_count*column_count, plane_count))
            data[index] = image_3d
            index = index + 1
            #For debuggin
            if DEBUG and index>9:
                break
        
        return data[:index]

    # We read the training and test set images and labels.
    X = load_images('/../datasets/afreightdata/data')
    Y = load_images('/../datasets/afreightdata/label')
   
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


X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()



def generateNoisySines( npoints, nseq, noise=0.3 ):
    """ construct a 2-class dataset out of noisy sines """
    x = np.arange(npoints)/float(npoints) * 20.
    y1 = np.sin(x+rand(1)*3.)
    y2 = np.sin(x/2.+rand(1)*3.)
   
    for _ in range(nseq):
        DS.newSequence()
        buf = rand(npoints)*noise + y1 + (rand(1)-0.5)*noise
        for i in range(npoints):
            DS.addSample([buf[i]],[0])
        DS.newSequence()
        buf = rand(npoints)*noise + y2 + (rand(1)-0.5)*noise
        for i in range(npoints):
            DS.addSample([buf[i]],[1])
    return DS

DS = SequenceClassificationDataSet

# create training and test data
trndata = SequenceClassificationDataSet(X_train, Y_train)
tstdata = SequenceClassificationDataSet(X_test, Y_test)


# construct LSTM network - note the missing output bias

rnn = buildNetwork( trndata.indim, (), trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outclass=SoftmaxLayer )

#buildNetwork( MultiDimensionalLSTM
#rnn.addInputModule(LinearLayer(3, name='in'))
#rnn.addModule(MDLSTMLayer(5,2, name='hidden'))
#rnn.addOutputModule(SoftmaxLayer(1, name='out'))
#
#rnn.addConnection(FullConnection(rnn['in'], rnn['hidden'], name='c1'))
#rnn.addConnection(FullConnection(rnn['hidden'], rnn['out'], name='c2'))
#
#rnn.addRecurrentConnection(FullConnection(rnn['hidden'], rnn['hidden'], name='c3'))
#rnn.sortModules()

# define a training method
trainer = RPropMinusTrainer(rnn, dataset=trndata, verbose=True )
# instead, you may also try
##trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001 )

# carry out the training
for i in range(100):
    trainer.trainEpochs( 2 )
    trnresult = 100. * (1.0-testOnSequenceData(rnn, trndata))
    tstresult = 100. * (1.0-testOnSequenceData(rnn, tstdata))
    print("train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult)

# just for reference, plot the first 5 timeseries
plot(trndata['input'][0:250,:],'-o')
hold(True)
plot(trndata['target'][0:250,0])
show()






