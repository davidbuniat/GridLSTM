# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from keras import activations, initializations
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from keras.layers.core import Layer, MaskedLayer
from six.moves import range
from keras.layers.recurrent import Recurrent

class GridLSTM(Recurrent):
    input_ndim = 4
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim, nb_rows, nb_cols, n_dim = 2,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, go_backwards=False, **kwargs):
        
        self.n_dim = n_dim + 1
        self.nb_cols = nb_cols
        self.nb_rows = nb_rows

        self.output_dim = 1 #output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards

        # Calculate the number of dimensions
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(GridLSTM, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[3]
        self.input = T.tensor4()

        self.W_i = self.init((self.n_dim, input_dim, self.output_dim))
        self.U_i = self.inner_init((self.n_dim, self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.n_dim, self.output_dim))

        self.W_f = self.init((self.n_dim, input_dim, self.output_dim))
        self.U_f = self.inner_init((self.n_dim, self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.n_dim, self.output_dim))

        self.W_c = self.init((self.n_dim, input_dim, self.output_dim))
        self.U_c = self.inner_init((self.n_dim, self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.n_dim, self.output_dim))

        self.W_o = self.init((self.n_dim, input_dim, self.output_dim))
        self.U_o = self.inner_init((self.n_dim, self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.n_dim, self.output_dim))

        self.params = [
            self.W_i, #self.U_i, self.b_i,
            self.W_c, #self.U_c, self.b_c,
            self.W_f, #self.U_f, self.b_f,
            self.W_o, #self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        
        h_t = o_t * self.activation(c_t)

        return h_t, c_t

    def LTSM(self, H, m, W_i, W_f, W_o, W_c):

        i_t = self.inner_activation(T.dot(W_i, H))
        f_t = self.inner_activation(T.dot(W_f, H))
        o_t = self.inner_activation(T.dot(W_o, H))
        c_t = self.activation(T.dot(W_c, H))

        m_t = f_t * m + i_t * c_t 
        h_t = self.activation(m_t * o_t) 
        return h_t, m_t 

    def step(self, x_t, H_x, H_y, M_x, M_y, W_i, W_f, W_o, W_c):
        
        #H_t = T.ones_like(H_x)
        #M_t = T.ones_like(H_x)
        #H = T.ones_like(H_x)

        H = T.stacklists([x_t, H_x[1], H_y[2]])
        M = T.stacklists([x_t, M_x[1], M_y[2]])
        for i in range(self.n_dim):  
            (H_temp, M_temp) = self.LTSM(H, M[i], W_i[i], W_f[i], W_o[i], W_c[i])

            if (i == 0):
                H_t = H_temp
                M_t = M_temp
            else:
                H_t = T.concatenate([H_t, H_temp], axis=0)
                M_t = T.concatenate([M_t, M_temp], axis=0)
        return H_t, M_t


    def get_output(self, train=False):
        X = self.get_input(train)
        #padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)

        #xi = T.dot(X, self.W_i) + self.b_i
        #xf = T.dot(X, self.W_f) + self.b_f
        #xc = T.dot(X, self.W_c) + self.b_c
        #xo = T.dot(X, self.W_o) + self.b_o

        output_model = dict(initial = theano.shared(
            np.zeros(((self.nb_rows+1)*(self.nb_cols), self.n_dim, 8, self.output_dim), dtype=np.float64)), 
            taps=[-1,-self.nb_cols])

        memory_model = dict(initial = theano.shared(
            np.zeros(((self.nb_rows+1)*(self.nb_cols), self.n_dim, 8, self.output_dim), dtype=np.float64)), 
            taps=[-1,-self.nb_cols])
        
        input_model = dict(input = T.flatten(X, outdim=3).dimshuffle((2, 0, 1)), taps=[self.nb_cols])

        [outputs, memories], updates = theano.scan(
            self.step,
            sequences= input_model,
            outputs_info=[
                output_model,
                memory_model
                ],
            non_sequences=[self.W_i, self.W_f, self.W_o, self.W_c],
            truncate_gradient=self.truncate_gradient,
            go_backwards=self.go_backwards,
            n_steps = self.nb_cols*(self.nb_rows-1))

        # Need to add padding 
#       print(outputs[2])
        output = outputs[0].dimshuffle(2,0,1)
        outputs = T.reshape(output, (8, self.output_dim, self.nb_rows, self.nb_cols))

        if self.return_sequences and self.go_backwards:
            return outputs[::-1].dimshuffle((1, 0, 2))
        elif self.return_sequences:
            return outputs[0]
        return outputs[-1]
    
    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.output_dim, self.nb_rows, self.nb_cols)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length,
                  "go_backwards": self.go_backwards}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

