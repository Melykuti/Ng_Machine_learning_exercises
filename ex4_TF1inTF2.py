'''
Neural networks. Forward propagation in an already trained network in `TensorFlow`. Computing the regularised cost function.
TensorFlow 1.x code run in TensorFlow 2.x via the compatibility mode.

There are independent choices in program flow:

Option 1 shows that you can have separate iterators for X and y
Option 2 has a single iterator for (X, y)

Option a processes single inputs (single images), takes 2.6 sec
Option b does batch processing of all images at once, takes 0.35 sec

Bence Mélykúti
09-19/03/2018, 31/01/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import time

# The network parameters are here for info, they are not actually used.
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading [and Visualizing] Data =============

data = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4data1.mat')
X = data['X']
y = data['y']
#y = y-1 # This transformation is not used.
y = y % 10 # Transforming 10 to 0, which is its original meaning.

# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

params = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4weights.mat')
#params = scipy.io.loadmat('../machine-learning-ex3/ex3/ex3weights.mat')
Theta1 = params['Theta1']   # Theta1 has size 25 x 401
Theta2 = params['Theta2']   # Theta2 has size 10 x 26

# To narrow computation to a subset of data for quick testing:
#X, y = X[1990:2010,:], y[1990:2010,:]

#  ================ Part 3: Compute Cost (Feedforward) ================

start_time = time.time()

tf.reset_default_graph() # not strictly necessary

batch_size = 5000 # try: X.shape[0] # 50 # 3900 # X.shape[0]-1

'''
## Option 1 with two individual iterators for X and y ##

## Option a ##
data_X = tf.data.Dataset.from_tensor_slices(X)
data_y = tf.data.Dataset.from_tensor_slices(y)

## Option b ##
#data_X = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
#data_y = tf.data.Dataset.from_tensor_slices(y).batch(batch_size)

x = data_X.make_one_shot_iterator().get_next() # https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
y_temp = data_y.make_one_shot_iterator().get_next()
## End of Option 1 ##

'''

## Option 2 with a single iterator for (X, y) ##

## Option a ##
#dataset = tf.data.Dataset.from_tensor_slices((X,y))
## Option b ##
dataset = tf.data.Dataset.from_tensor_slices((X,y)).batch(batch_size)

# These two (with and without brackets) are equally valid:
x, y_temp = dataset.make_one_shot_iterator().get_next()
#(x, y_temp) = dataset.make_one_shot_iterator().get_next() # https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
# https://www.tensorflow.org/programmers_guide/datasets#consuming_values_from_an_iterator
## End of Option 2 ##


net1 = tf.layers.dense(tf.reshape(x, [-1,X.shape[1]]), units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0])) # the tf.reshape is needed if batch size = 1 (there is no .batch())

# activation=tf.nn.sigmoid would be wrong because the loss function turns the logits values into sigmoid(logits), so there must be no application of sigmoid in the output of this layer:
logits = tf.layers.dense(net1, units=Theta2.shape[0], use_bias=True, activation=None, kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0]))

# Rearranging the columns: digits [1,2,...,9,0] are mapped to digits [0,1,2,...,9], that is, column 9 (digit 0, encoded with position 10 of [1,10]) must come first, the rest must be shifted up by one.
logits = tf.gather(logits, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)],0), axis=1)
#logits = tf.gather(logits, np.concatenate(([9], np.arange(0,9))), axis=1) # equivalent, with np arrays

# Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
y_idcc = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
y_onehot = tf.feature_column.indicator_column(y_idcc)
y_layer = tf.feature_column.input_layer({'labels': y_temp}, y_onehot)
loss = tf.losses.sigmoid_cross_entropy(y_layer, logits) * 10
#loss = tf.losses.sigmoid_cross_entropy(tf.cast(y_layer, tf.float64), logits) * 10 # Is tf.cast even needed here?
# tf.losses.sigmoid_cross_entropy already produces a scalar, tf.reduce_mean doesn't do anything with it:
#loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.cast(y_layer, tf.float64), logits)) * 10
# This also works:
#loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_layer, tf.float64), logits=logits, reduction=tf.losses.Reduction.NONE)) * 10

# This instead of the last two lines (y_layer = , loss = ) doesn't work:
#y_layer = tf.feature_column.input_layer({'labels': tf.cast(y_temp,tf.float64)}, y_onehot)
#loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_layer, logits)) * 10

# This instead of loss = doesn't work:
#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_layer, logits=logits)
# But this variant works:
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_layer, tf.float64), logits=logits)) * 10

# Selecting which digit

# For individual processing:
#digit = (tf.argmax(logits[0], axis=0)+1) % 10 # if there is no rearrangement of columns by tf.gather, digits are stored as [1,2,...,9,0]
#digit = tf.argmax(logits[0], axis=0) # if columns are rearranged by tf.gather, digits are stored as [0,1,2,...,9]

# For batch processing:
#digit = tf.map_fn(lambda x: (tf.argmax(x, axis=0)+1) % 10, logits, dtype=tf.int64) # if there is no rearrangement of columns by tf.gather, digits are stored as [1,2,...,9,0]
digit = tf.map_fn(lambda x: tf.argmax(x, axis=0), logits, dtype=tf.int64) # if columns are rearranged by tf.gather, digits are stored as [0,1,2,...,9]


# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.

lambda0 = 1
# loss with penalty:
losswp = loss + lambda0 * 0.5 * (np.sum(Theta1[:,1:]*Theta1[:,1:], axis=None) + np.sum(Theta2[:,1:]*Theta2[:,1:], axis=None)) / batch_size



with tf.Session() as sess:
    #tf.global_variables_initializer()
    sess.run(tf.global_variables_initializer())

## Option b ##

    ll, llwp, dd = sess.run([loss, losswp, digit])
    pred=dd.reshape(-1,1)

## End of Option b ##

## Option a ##
    '''
    pred = -np.ones((X.shape[0],1), dtype=int)
    ll=0

    for i in range(X.shape[0]):
        loss_incr, losswp_incr, dd = sess.run([loss, losswp, digit])
        ll=ll + loss_incr/X.shape[0]
        pred[i,0] = dd
    llwp = ll + lambda0 * 0.5 * (np.sum(Theta1[:,1:]*Theta1[:,1:], axis=None) + np.sum(Theta2[:,1:]*Theta2[:,1:], axis=None)) / X.shape[0]
    '''
## End of Option a ##


    # This shows that we actually reach the end of dataset:
    # https://www.tensorflow.org/programmers_guide/datasets#consuming_values_from_an_iterator
    try:
        print('Loss on the next batch: {0:.6f}.'.format(sess.run(loss)))
    except tf.errors.OutOfRangeError:
        print("End of dataset reached.")

print('\nCost at parameters (loaded from ex4weights): {0:.6f}.'.format(ll))
print('Expected loss (approx.): 0.287629.')

print('\nRegularised cost at parameters (loaded from ex4weights): {0:.6f}.'.format(llwp))
print('Expected regularised loss (approx.): 0.383770.')

print('\nTraining Set Accuracy: {0:.2f}%.'.format(np.mean(pred == y[:len(pred)]) * 100))
print('Expected training accuracy on complete Training Set (approx.): 97.5%.')

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))

