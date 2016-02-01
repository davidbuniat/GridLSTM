from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,TimeDistributedDense, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD,Adam
import data as dt
import gridLstm as gl

'''
    Train a Bidirectional LSTM on the IMDB sentiment classification task.
    GPU command:

      THEANO_FLAGS='cuda.root=/Developer/NVIDIA/CUDA-7.5,device=gpu,floatX=float32' python train.py
    Output after 4 epochs on CPU: ~0.8146
    Time per epoch on CPU (Core i7): ~150s.
'''

max_features = 3
maxlen = 19200  # cut texts after this number of words (among top max_features most common words)
batch_size = 10
nb_epoch = 1
row_count = 120
column_count = 160
number_of_training_data = 455
DEBUG = False 

N_HIDDEN = 20
X_train, y_train, X_val, y_val, X_test, y_test = dt.load_dataset(number_of_training_data, row_count, column_count, max_features, DEBUG)
dt.saveImage(y_train[0], "gt.png", row_count, column_count,  3, True)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
#X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)
print('Build model...')



model = Graph()
model.add_input(name='input', input_shape=(None,max_features), dtype='float')
#model.add_node(Embedding(3, 128, input_length=(maxlen, 3)),name='embedding', input='input')
#model.add(TimeDistributedDense(2, init='uniform', input_dim=3), name = 'input')

model.add_node(gl.GridLSTM(N_HIDDEN, return_sequences=True), name='forward', input='input')
#model.add_node(LSTM(N_HIDDEN, return_sequences=True), name='forward_2', input='forward')
#model.add_node(LSTM(N_HIDDEN, return_sequences=True), name='forward_3', input='forward_2')
#model.add_node(LSTM(N_HIDDEN, return_sequences=True), name='forward_4', input='forward_3')
#model.add_node(LSTM(N_HIDDEN, return_sequences=True), name='forward_5', input='forward_4')
#model.add_node(LSTM(N_HIDDEN, return_sequences=True, go_backwards=True ), name='backward', input='input')
model.add_node(Dropout(0.5), name='dropout', input='forward')
#model.add_node(Merge(layers=['forward', 'backward'], mode='sum'))
#model.add_node(Merge(layers=['forward', 'backward'], mode='sum'))
#model.add_node(TimeDistributedDense(output_dim = 10, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_node(TimeDistributedDense(output_dim = 1, activation='sigmoid'), name='softmax', input='dropout')
model.add_output(name='output', input='softmax')

# try using different optimizers and different optimizer configs
Adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#model.compile(optimizer=Adam, loss='mean_squared_error' )
model.compile(Adam, {'output': 'mean_squared_error'})

print("Train...")
model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=nb_epoch)


acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))

#model.compile(loss='mean_squared_error', optimizer='sgd')

inp = X_test[1]
gt = y_test[1]
out = np.round(np.array(model.predict({'input': X_test},batch_size=batch_size)['output']))[0]
#dt.saveImage(inp, "input.png", row_count, column_count,  3, False)
dt.saveImage(gt, "gt.png", row_count, column_count,  3, True)
dt.saveImage(out, "output.png", row_count, column_count,  3, True)
print('Test accuracy:', acc)