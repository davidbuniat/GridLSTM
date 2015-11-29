from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.

from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork

#from .datasets import generateNoisySines

import numpy as np
from numpy.random import multivariate_normal, rand
from scipy import diag
from pylab import show, hold, plot

def generateNoisySines( npoints, nseq, noise=0.3 ):
    """ construct a 2-class dataset out of noisy sines """
    x = np.arange(npoints)/float(npoints) * 20.
    y1 = np.sin(x+rand(1)*3.)
    y2 = np.sin(x/2.+rand(1)*3.)
    DS = SequenceClassificationDataSet(1,1, nb_classes=2)
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

# create training and test data
trndata = generateNoisySines(50, 40)
trndata._convertToOneOfMany( bounds=[0.,1.] )
tstdata = generateNoisySines(50, 20)
tstdata._convertToOneOfMany( bounds=[0.,1.] )

# construct LSTM network - note the missing output bias
rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

# define a training method
trainer = RPropMinusTrainer( rnn, dataset=trndata, verbose=True )
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

