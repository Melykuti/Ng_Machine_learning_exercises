'''
Neural networks. Forward propagation in an already trained network in TensorFlow 2.0. Computing the regularised cost function.

exec(open('ex4_TF2_evaluate.py').read())

TF 2.0:
sigmoid_step_option 0-4 all take 0.45-0.5 sec.

Bence Mélykúti
09-19/03/2018, 31/01-07/02, 28/02-02/03/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

### User input ###

sigmoid_step_option = 0 # {0, 1, 2, 3, 4}
metric_choice_option = 0 # {0, 1} passes in a string or a class, respectively
loss_regularization_option = 0 # {0, 1, 2}
batch_size = 5000 # try: 50 or 3900 or 4999 or 5000 (which is X.shape[0]) or 50000.
# It seems that when batch_size is not a divisor of 5000, then the loss and regularised loss are incorrect.

### End of input ###

# The network parameters are here for info, they are not actually used.
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading [and Visualizing] Data =============

data = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4data1.mat')
X = data['X']
y = data['y']
y = y % 10 # Transforming 10 to 0, which is its original meaning.

# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

params = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4weights.mat')
Theta1 = params['Theta1']   # Theta1 has size 25 x 401
Theta2 = params['Theta2']   # Theta2 has size 10 x 26

# To narrow computation to a subset of data for quick testing:
#X, y = X[1990:2010,:], y[1990:2010,:]

#  ================ Part 3: Compute Cost (Feedforward) ================

tf.keras.backend.clear_session() # not strictly necessary

start_time = time.time()

# Both options for l0 are good:
l0 = tf.keras.layers.Dense(Theta1.shape[0], use_bias=True, activation='sigmoid', kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]))
#l0 = tf.keras.layers.Dense(Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]))

# No activation is specified, meaning we get the identity function as activation.
l1 = tf.keras.layers.Dense(Theta2.shape[0], use_bias=True,
    kernel_initializer=tf.constant_initializer(Theta2[:,1:].T),
    bias_initializer=tf.constant_initializer(Theta2[:,0]))

# With tf.gather, I permute the 10 columns: digits [1,2,...,9,0] are mapped to digits [0,1,2,...,9], that is, column 9 (digit 0, encoded with position 10 of [1, 2, ... 10]) must come first, the rest must be shifted up by one.
if sigmoid_step_option in [0, 3]: # We apply sigmoid function.

    # Both options work:
    l1.activation=lambda x: tf.gather(tf.sigmoid(x), tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)
    #l1.activation=lambda x: tf.sigmoid(tf.gather(x, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1))

# ...tf.gather(x, np.concatenate(([9], np.arange(0,9))), axis=1)... would do the same with np arrays.


else: # sigmoid_step_option in [1, 2, 4]; in these cases sigmoid function is not yet applied.

    # Here activation=tf.nn.sigmoid would be wrong because l2 (in sigmoid_step_option==1) or
    # the loss function (in sigmoid_step_option==2) turns the logits values into sigmoid(output),
    # so there must be no application of sigmoid in the output of this layer.
    l1.activation=lambda x: tf.gather(x, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)


if sigmoid_step_option != 1:
    layers_model = [l0, l1]
else: # sigmoid_step_option=1
    # Without specifying kernel_initializer to be the identity matrix, it would multiply with a random matrix!
    l2 = tf.keras.layers.Dense(Theta2.shape[0], activation='sigmoid', kernel_initializer=tf.constant_initializer(np.eye(Theta2.shape[0])), use_bias=False)
    layers_model = [l0, l1, l2]
# For sigmoid_step_option in [0, 1, 3], pred has been fed through a sigmoid function, it's in [0; 1].
# For sigmoid_step_option in [2, 4], pred has not been fed through a sigmoid function, it's in ]-infty; infty[.

model = tf.keras.Sequential(layers_model)

'''
# Version for TensorFlow 1.x
y_idcc = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
y_onehot = tf.feature_column.indicator_column(y_idcc)
y_layer = tf.feature_column.input_layer({'labels': y_temp}, y_onehot)
loss = tf.losses.sigmoid_cross_entropy(y_layer, logits) * 10
'''
# y needs to be turned into one hot encoding
l10 = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
l11 = tf.feature_column.indicator_column(l10)
l12 = tf.keras.layers.DenseFeatures(l11)
layers_y = [l12]
model_y = tf.keras.Sequential(layers_y)
y_onehot = model_y.predict({'labels': y})
dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32),
            y_onehot.astype(np.float32))).batch(batch_size)

if sigmoid_step_option in [0, 1, 3]:
    if metric_choice_option==0:
        model.compile(optimizer='adam',
                      #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['categorical_accuracy'])
                      #metrics=['binary_crossentropy']) # With this, accuracy will be the same as loss.
                     # But we probably can't set loss to receive accuracy, there's no such loss function.
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
else:
    if metric_choice_option==0:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['categorical_accuracy'])
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

# These don't work:
#dataset_X = dataset.map(lambda x, y: x); dataset_y = dataset.map(lambda x, y: y)
#loss, accuracy = model.evaluate(dataset_X)
#loss, accuracy = model.evaluate(x=dataset_X, y=dataset_y)
#loss, accuracy = model.evaluate(x=dataset_X, y=y_onehot)
# This does work:
loss, accuracy = model.evaluate(dataset)

# Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
loss = 10*loss

#print("Accuracy", accuracy, ". Loss", loss)

# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.

lambda0 = 1
# losswp = loss with penalty from regularisation

if loss_regularization_option == 0:
    losswp = loss + lambda0 * 0.5 * (np.sum(Theta1[:,1:]*Theta1[:,1:], axis=None) + np.sum(Theta2[:,1:]*Theta2[:,1:], axis=None)) / len(y)

elif loss_regularization_option == 1:
    losswp = loss + lambda0 * 0.5 * tf.math.reduce_sum(tf.math.square(tf.constant(Theta1[:,1:], dtype=tf.float32))) / len(y)\
                  + lambda0 * 0.5 * tf.math.reduce_sum(tf.math.square(tf.constant(Theta2[:,1:], dtype=tf.float32))) / len(y)

else:
    regularizer = tf.keras.regularizers.l2(lambda0 * 0.5)
    losswp = loss + tf.cast(regularizer(Theta1[:,1:]), dtype=tf.float32) / len(y)\
                  + tf.cast(regularizer(Theta2[:,1:]), dtype=tf.float32) / len(y)

print('\nCost at parameters (loaded from ex4weights): {0:.6f}.'.format(loss))
print('Expected loss (approx.): 0.287629.')

print('\nRegularised cost at parameters (loaded from ex4weights): {0:.6f}.'.format(losswp))
print('Expected regularised loss (approx.): 0.383770.')

print('\nTraining Set Accuracy: {0:.2f}%.'.format(accuracy * 100))
print('Expected training accuracy on complete Training Set (approx.): 97.5%.')

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))

