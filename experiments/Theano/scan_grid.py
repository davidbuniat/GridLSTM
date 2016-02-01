import theano
import theano.tensor as T
import numpy as np
theano.config.warn.subtensor_merge_bug = False

i = T.iscalar("i")
x = T.iscalar("x")
y = T.iscalar("y")
A = T.imatrix("A") 

def inner_sum(prior_x, B):
	return prior_x+B

def inner_sum2D(x_t, y_t,u):
	return x_t + y_t + u

row_count = 3
column_count = 4

# Symbolic description of the result
result, updates = theano.scan(fn=inner_sum2D,
							sequences = dict(input = T.flatten(A), taps=[column_count]), 
                            outputs_info = dict(initial = T.flatten(A), taps=[-1,-column_count]),
                            #non_sequences= 
                            n_steps=x*y)

# Scan has provided us with A ** 1 through A ** k.  Keep only the last
# value. Scan notices this and does not waste memory saving them.
final_result = T.reshape(result, (x,y))

sum_theano = theano.function(inputs=[A, x, y], outputs=final_result,
                      updates=updates)

#Need to add one row of pixels for padding
#Need to decide about the padding at the edges
img = [[0,0,0,0],[1,1,0,0],[0,0,0,0],[0,0,1,0]]

img = np.reshape(img, (column_count, row_count+1))
print(np.reshape(sum_theano(img,column_count,row_count), (row_count, column_count)))