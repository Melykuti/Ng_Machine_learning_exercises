'''
Backpropagation and training a neural network by fitting a Keras model.
https://www.tensorflow.org/guide/keras/overview

f=open('ex4_TF2_Keras.py','r'); exec(f.read()); f.close()

- Can I extract the edge weights from the neural network?
Yes, from an already initialised layer before the training has started, or once the training has completed.
https://www.tensorflow.org/api_docs/python/tf/keras/Model#get_layer
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_weights

- Can I add regularisation?
Yes, one can add regularisation (of kernel or bias or output) to each layer of the neural network. This will enter the model's loss function.
https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer

Bence Mélykúti
09-17/03/2018, 03/03/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

### User input ###

epochs = 5 # number of epochs in training
logits_option = 1 # {0, 1}, 0 for from_logits=False and tf.sigmoid in Layer 1; 1 for True and no sigmoid
# "Using from_logits=True may be more numerically stable."
deterministic_initialisation = 1 # {0, 1}
regularization = 0 # {0, 1}
batch_size = 5000 # try: 50 or 3900 or 4999 or 5000 (which is X.shape[0]) or 50000

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

#  ================ Part 3: Compute Cost (Feedforward) ================

tf.keras.backend.clear_session() # not strictly necessary

def penalty_from_regularisation(model, lambda0, denom):
    pen = 0
    for i in range(2):
        # Both versions for Theta are correct.
        Theta = model.get_weights()[2*i]
        #Theta = model.get_layer(index=i).get_weights()[0]
        pen += lambda0 * 0.5 * np.sum(Theta*Theta, axis=None) / denom
    return pen

start_time = time.time()

if deterministic_initialisation == 1:
    l0 = tf.keras.layers.Dense(Theta1.shape[0], use_bias=True, activation='sigmoid',
        kernel_initializer=tf.constant_initializer(Theta1[:,1:].T),
        bias_initializer=tf.constant_initializer(Theta1[:,0]), name='layer_0')

    # No activation is specified, meaning we get the identity function as activation.
    l1 = tf.keras.layers.Dense(Theta2.shape[0], use_bias=True,
        kernel_initializer=tf.constant_initializer(Theta2[:,1:].T),
        bias_initializer=tf.constant_initializer(Theta2[:,0]), name='layer_1')
else: # random initialisation
    l0 = tf.keras.layers.Dense(Theta1.shape[0], use_bias=True, activation='sigmoid',
        kernel_initializer=tf.keras.initializers.he_uniform(),
        bias_initializer=tf.keras.initializers.he_uniform(), name='layer_0')

    # No activation is specified, meaning we get the identity function as activation.
    l1 = tf.keras.layers.Dense(Theta2.shape[0], use_bias=True,
        kernel_initializer=tf.keras.initializers.he_uniform(),
        bias_initializer=tf.keras.initializers.he_uniform(), name='layer_1')

lambda0 = 1
if regularization == 1:
    # Divide lambda0 by 10 to counteract the multiplication of loss by 10 later on.
    l0.kernel_regularizer=tf.keras.regularizers.l2(0.1*lambda0 * 0.5 / len(y))
    l0.bias_regularizer=None
    l1.kernel_regularizer=tf.keras.regularizers.l2(0.1*lambda0 * 0.5 / len(y))
    l1.bias_regularizer=None

if logits_option == 1:
    l1.activation = lambda x: tf.gather(x, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)
else:
    l1.activation = lambda x: tf.gather(tf.sigmoid(x), tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)


layers_model = [l0, l1]
model = tf.keras.Sequential(layers_model)

# y needs to be turned into one hot encoding
l10 = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
l11 = tf.feature_column.indicator_column(l10)
l12 = tf.keras.layers.DenseFeatures(l11)
layers_y = [l12]
model_y = tf.keras.Sequential(layers_y)
y_onehot = model_y.predict({'labels': y})

dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32),
            y_onehot.astype(np.float32))).batch(batch_size)
dataset_shuffled = tf.data.Dataset.from_tensor_slices((X.astype(np.float32),
            y_onehot.astype(np.float32))).shuffle(len(y), reshuffle_each_iteration=True)\
            .batch(batch_size)

if logits_option == 1:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['categorical_accuracy'])
else:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['categorical_accuracy'])

loss_orig, accuracy_orig = model.evaluate(dataset)
# Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
loss_orig = 10*loss_orig
if regularization == 0:
    losswp_orig = loss_orig + penalty_from_regularisation(model, lambda0, len(y))

model.fit(dataset_shuffled, epochs=epochs)
# This doesn't work:
#dataset_X = dataset.map(lambda x, y: x); dataset_y = dataset.map(lambda x, y: y)
#model.fit(dataset_X, dataset_y, epochs=epochs)

loss, accuracy = model.evaluate(dataset)
# Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
loss = 10*loss
if regularization == 0:
    losswp = loss + penalty_from_regularisation(model, lambda0, len(y))

if regularization == 0:
    if deterministic_initialisation == 1:
        print('\nExpected loss at parameters loaded from ex4weights.mat (approx.): 0.287629.')
    else:
        print()
    print('Loss before the start of training: {0:.6f}.'.format(loss_orig))
    print('Loss after training: {0:.6f}.'.format(loss))

if deterministic_initialisation == 1:
    print('\nExpected regularised loss at parameters loaded from ex4weights.mat (approx.):\n0.383770.')
else:
    print()
if regularization == 0:
    print('Manually computed regularised loss before the start of training: {0:.6f}.'.format(losswp_orig))
    print('Manually computed regularised loss after training: {0:.6f}.'.format(losswp))
else:
    print('Regularised loss before the start of training: {0:.6f}.'.format(loss_orig))
    print('Regularised loss after training: {0:.6f}.'.format(loss))


if deterministic_initialisation == 1:
    print('\nOriginal training accuracy on complete Training Set (approx.): 97.5%.')
else:
    print()
print('Training Set Accuracy before the start of training: {0:.2f}%.'.format(accuracy_orig * 100))
print('Training Set Accuracy after training: {0:.2f}%.'.format(accuracy * 100))

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))


'''
Different ways of accessing the layers' kernels, biases:
model.get_weights()[0].shape==(400,25)
model.get_weights()[1].shape==(25,)
model.get_weights()[2].shape==(25,10)
model.get_weights()[3].shape==(10,)

model.get_layer('layer_0').get_weights()[0].shape==(400,25)
model.get_layer('layer_0').get_weights()[1].shape==(25,)
model.get_layer('layer_1').get_weights()[0].shape==(25,10)
model.get_layer('layer_1').get_weights()[1].shape==(10,)

model.get_layer(index=0).get_weights()[0].shape==(400,25)
model.get_layer(index=0).get_weights()[1].shape==(25,)
model.get_layer(index=1).get_weights()[0].shape==(25,10)
model.get_layer(index=1).get_weights()[1].shape==(10,)
'''

'''
# Extracting neural network parameters to store in Theta1, Theta2 matrices:
Theta1=np.zeros((25,401))
Theta1[:,0] = model.get_layer(index=0).get_weights()[1]
Theta1[:,1:] = model.get_layer(index=0).get_weights()[0].T
Theta2=np.zeros((10,26))
Theta2[:,0] = model.get_layer(index=1).get_weights()[1]
Theta2[:,1:] = model.get_layer(index=1).get_weights()[0].T

# These can now be passed into ex4_TF1inTF2.py or ex4_TF2.py or ex4_TF2_evaluate.py, just comment out the definitions of Theta1 and Theta2 in that file to avoid overwriting them.
'''
