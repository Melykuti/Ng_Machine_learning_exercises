'''
Neural networks. Forward propagation in an already trained network in TensorFlow (to use the network for classification).
TensorFlow 1.x code run in TensorFlow 2.x via the compatibility mode.

Be careful: 
According to tf.layers.dense documentation (https://www.tensorflow.org/api_docs/python/tf/layers/dense):
outputs = activation(inputs*kernel + bias)
The kernel matrix multiplies from right! (And the inputs are seen as a row vector.) This is why I have to transpose the loaded network parameters Theta1 and Theta2.


There are independent choices in program flow:

Option 1 is with tf.layers.Input()
Option 2 is without tf.layers.Input()

Original timing with TF 1.3 or 1.5 (timing with TF 2.1):
Option a processes single inputs (single images), takes 1.5 sec (2.5-2.8 sec with TF 2.1)
Option b does batch processing of all images at once, takes 0.3 sec (0.6 sec with TF 2.1)

Bence Mélykúti
09-19/03/2018, 29-30/01/2020
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

data = scipy.io.loadmat('../machine-learning-ex3/ex3/ex3data1.mat')
X = data['X']
y = data['y']
y = y % 10 # Transforming 10 to 0, which is its original meaning.

# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load the pre-initialized 
# neural network parameters.

params = scipy.io.loadmat('../machine-learning-ex3/ex3/ex3weights.mat')
Theta1 = params['Theta1']   # Theta1 has size 25 x 401
Theta2 = params['Theta2']   # Theta2 has size 10 x 26

start_time = time.time()

# ================= Part 3: Implement Predict =================
#  After training a neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

tf.reset_default_graph() # Not strictly necessary, but good to see this call once. This resets the graph, which is useful if (and only if) you have already run the network but something went wrong and when you try to run it again, you get some hard to understand error message.


# Difference between tf.data.Dataset.from_tensors and tf.data.Dataset.from_tensor_slices: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
# from_tensors reads all data at once; from_tensor_slices reads line by line, which is preferable for huge datasets
# With from_tensors, you'd also need to pull out each row from the tensor somehow.
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
# https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays

# To narrow computation to a subset of data for quick testing:
#X, y = X[1990:2010,:], y[1990:2010,:]

#dataset = tf.data.Dataset.from_tensors((X,y))
#dataset = tf.data.Dataset.from_tensors(X)
#dataset = tf.data.Dataset.from_tensor_slices((X,y))
#dataset = tf.reshape(tf.data.Dataset.from_tensor_slices(X), [-1,X.shape[1]])

## Option a ##
#dataset = tf.data.Dataset.from_tensor_slices(X)
## Option b ##
dataset = tf.data.Dataset.from_tensor_slices(X).batch(X.shape[0])


'''
# There are a lot of ways to do this wrong. Some examples of what doesn't work:

#x = tf.get_variable(x, shape=(X.shape[1],1))
#x = tf.constant(X[0,:], shape=(X.shape[1],1))
#x = tf.placeholder(tf.float64, shape=(X.shape[1],1))
#net = tf.layers.dense(tf.reshape(x, [X.shape[1], 1]), units=X.shape[1], use_bias=True, kernel_initializer=Theta1[:,1:], bias_initializer=Theta1[:,0])
# ValueError: Input 0 of layer dense_1 is incompatible with the layer: : expected min_ndim=2, found ndim=1. Full shape received: [400]

#x, ytemp = dataset.make_one_shot_iterator().get_next()
#x = np.reshape(dataset.make_one_shot_iterator().get_next(),[1,-1])
#x = dataset.make_one_shot_iterator().get_next()
#x2 = tf.Variable(x) # tf.as_dtype(

#net = tf.layers.dense(tf.reshape(x, [1,X.shape[1]]), units=X.shape[1], activation=None)
#net = tf.layers.dense(tf.reshape(x, [1,X.shape[1]]), units=X.shape[1], activation=None, use_bias=True, kernel_initializer=Theta1[:,1:], bias_initializer=Theta1[:,0])
#net = tf.layers.dense(tf.reshape(dataset.make_one_shot_iterator().get_next(), [X.shape[1],1]), units=X.shape[1], use_bias=True, kernel_initializer=Theta1[1:], bias_initializer=Theta1[0])
#net = tf.layers.dense(tf.reshape(x, [1,X.shape[1]]), units=X.shape[1], use_bias=True, kernel_initializer=Theta1[:,1:], bias_initializer=Theta1[:,0])
#net = tf.layers.dense(tf.reshape(net,[1,-1]), units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=Theta1[:,1:], bias_initializer=Theta1[:,0])
#net = tf.layers.dense(tf.reshape(x, [1,X.shape[1]]), units=X.shape[1], use_bias=True, kernel_initializer=assign_from_values_fn(Theta1[:,1:]), bias_initializer=assign_from_values_fn(Theta1[:,0]))
#net = tf.layers.dense(x[:,np.newaxis], units=X.shape[1], use_bias=True, kernel_initializer=Theta1[1:], bias_initializer=Theta1[0])
#net = tf.layers.dense(net, units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=Theta2[:,1:], bias_initializer=Theta2[:,0])
#net = tf.layers.dense(net, units=X.shape[1], use_bias=True, kernel_initializer=Theta1[:,1:], bias_initializer=Theta1[:,0])
#logits = tf.layers.dense(net, units=Theta2.shape[0], activation=None)
'''

x = dataset.make_one_shot_iterator().get_next() # https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
'''
## This also works:
iter=dataset.make_one_shot_iterator()
x=iter.get_next()
'''

'''
## Option 1, with tf.layers.Input() ##
# https://www.tensorflow.org/api_docs/python/tf/layers/Input
# shape: A shape tuple (integer), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
# all work:
#net = tf.layers.Input(shape=[1,X.shape[1]], tensor=tf.reshape(x, [1,X.shape[1]]))
#net = tf.layers.Input(shape=0, tensor=tf.reshape(x, [1,X.shape[1]]))
net = tf.layers.Input(tensor=tf.reshape(x, [-1,X.shape[1]]))

net1 = tf.layers.dense(net, units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]))
## End of Option 1 ##

'''

## Option 2, without tf.layers.Input() ##
net1 = tf.layers.dense(tf.reshape(x, [-1,X.shape[1]]), units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]))
# doesn't work:
#net = tf.layers.Input(shape=[1,X.shape[1]], tensor=x)
#net = tf.layers.Input(shape=X.shape[1], tensor=x)
## End of Option 2 ##


# Both Options 1 and 2 continue with this:
logits = tf.layers.dense(net1, units=Theta2.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0]))

# The output layer (logits) has 10 units, for digits 1,2,...,9,0. After taking argmax, you have to map the result of argmax, 0,1,2,...,9 to the required 1,2,...,9,0.
## Individual processing for Option a: ##
#digit = (tf.argmax(logits[0], axis=0)+1) % 10 # selecting which digit
## Batch processing for either Option a or Option b: ##
digit = tf.map_fn(lambda x: (tf.argmax(x, axis=0)+1) % 10, logits, dtype=tf.int64)

with tf.Session() as sess:
    #tf.global_variables_initializer()
    sess.run(tf.global_variables_initializer())

## Option b ##
    
    dd = sess.run(digit)
    #ll, dd = sess.run([logits, digit])
    pred=dd.reshape(-1,1)
    

## Option a ##
    '''
    pred = -np.ones((X.shape[0],1), dtype=int)
    for i in range(X.shape[0]):
        dd = sess.run(digit)
        #dd = digit.eval()
        pred[i,0] = dd
    '''

    # This shows that we actually reach the end of dataset:
    # https://www.tensorflow.org/programmers_guide/datasets#consuming_values_from_an_iterator
    try:
        print(sess.run(digit))
    except tf.errors.OutOfRangeError:
        print("End of dataset")


print('\nTraining Set Accuracy: {0:.2f}%.'.format(np.mean(pred == y) * 100))
print('Expected training error value on complete Training Set (approx.): 97.5%.')

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))

plt.scatter(np.arange(len(y)), y, label='Ground truth')
plt.scatter(np.arange(len(y)), pred, marker=".", c='r', label='Prediction')
plt.xlabel('Sample ID')
plt.ylabel('Digit')
plt.legend()
plt.show()

