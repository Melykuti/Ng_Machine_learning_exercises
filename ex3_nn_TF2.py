'''
Neural networks. Forward propagation in an already trained network in TensorFlow 2.0-2.1 (to use the network for classification).

TF 2.1: option==2, 3, 4, 5 work; options 0, 1 and 6 fail with "AttributeError: 'RepeatedCompositeFieldContainer' object has no attribute 'append'" (But mine hasn't installed properly.)
Option 2 takes 5.7-6.1 sec.
Option 3 takes 3.1 sec.
Option 4 takes 5.7-6 sec.
Option 5 takes 1.8 sec.
Option 7 takes 1.8 sec.
TF 2.0:
Option 0 takes 1.75 sec.
Option 1 takes 1.75 sec.
Option 6 takes 1.8 sec.
Option 2 takes 6.1 sec.
Option 3 takes 3.1 sec.
Option 4 takes 6.3 sec.
Option 5 takes 1.8 sec.
Option 7 takes 1.8 sec.

Be careful: 

According to tf.keras.layers.Dense (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense):
output = activation(dot(input, kernel) + bias)
The kernel matrix multiplies from right! (And the inputs are seen as a row vector.) This is why I have to transpose the loaded network parameters Theta1 and Theta2.

# According to tf.layers.dense documentation (https://www.tensorflow.org/api_docs/python/tf/layers/dense):
outputs = activation(inputs*kernel + bias)

[In version for Tensorflow 1.x, there used to be two independent choices in program flow:

Option 1 is with tf.layers.Input()
Option 2 is without tf.layers.Input()

Option a processes single inputs (single images), takes 1.5 sec
Option b does batch processing of all images at once, takes 0.3 sec
]

Bence Mélykúti
09-19/03/2018, 27-30/1/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

### User input ###

option = 0 # {0, 1, ..., 7}

### End of input ###

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

tf.keras.backend.clear_session()
start_time = time.time()

# ================= Part 3: Implement Predict =================
#  After training a neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.


# Difference between tf.data.Dataset.from_tensors and tf.data.Dataset.from_tensor_slices: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
# from_tensors reads all data at once; from_tensor_slices reads line by line, which is preferable for huge datasets
# With from_tensors, you'd also need to pull out each row from the tensor somehow.
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
# https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays

# To narrow computation to a subset of data for quick testing:
#X, y = X[1990:2010,:], y[1990:2010,:]

if option==2 or option==3:
    dataset = tf.data.Dataset.from_tensor_slices(X)
else:
    dataset = tf.data.Dataset.from_tensor_slices(X).batch(X.shape[0])
    #dataset = tf.data.Dataset.from_tensor_slices(X).batch(64) # this is about the same speed as .batch(X.shape[0])
    #dataset = tf.data.Dataset.from_tensor_slices(X).batch(1) # this also works but it is 1.5x-4x slower


# It also works with tf.keras.initializers.Constant() in place of tf.constant_initializer because these are only aliases: https://www.tensorflow.org/api_docs/python/tf/constant_initializer .

if option==0:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(Theta1.shape[0], activation='sigmoid', use_bias=True, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]), input_shape=[X.shape[1]]))
    model.add(tf.keras.layers.Dense(Theta2.shape[0], activation='sigmoid', use_bias=True, kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0])))  # One doesn't even need the second sigmoid activation function because it is monotone increasing and doesn't change the ordering for argmax.

    pred = model.predict(dataset)

elif option==1:
    # input_shape=[X.shape[1]] could be left out below
    layers = [tf.keras.layers.Dense(Theta1.shape[0], kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]), activation='sigmoid', input_shape=[X.shape[1]]),
             tf.keras.layers.Dense(Theta2.shape[0], kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0]), activation='sigmoid')]  # One doesn't even need the second sigmoid activation function because it is monotone increasing and doesn't change the ordering for argmax.

    # This doesn't work as tf.constant_initializer() doesn't take Tensors as input.
    #layers = [tf.keras.layers.Dense(Theta1.shape[0], kernel_initializer= tf.constant_initializer(tf.transpose(Theta1[:,1:])), bias_initializer=tf.constant_initializer(Theta1[:,0]), activation='sigmoid'),
    #    tf.keras.layers.Dense(Theta2.shape[0], kernel_initializer= tf.constant_initializer(tf.transpose(Theta2[:,1:])), bias_initializer=tf.constant_initializer(Theta2[:,0]), activation='sigmoid')]

    # This doesn't work: ValueError: Could not interpret initializer identifier: tf.Tensor(...)
    #layers = [tf.keras.layers.Dense(Theta1.shape[0], kernel_initializer=tf.transpose(Theta1[:,1:]), bias_initializer=Theta1[:,0], activation='sigmoid'),
    #    tf.keras.layers.Dense(Theta2.shape[0], kernel_initializer=tf.transpose(Theta2[:,1:]), bias_initializer=Theta2[:,0], activation='sigmoid')]

    model = tf.keras.Sequential(layers)
    #model = tf.keras.models.Sequential(layers) # This is just an alias of previous.

    #model.build() # not necessary

    pred = model.predict(dataset)

elif option==6:

    class NNModel(tf.keras.Model):

      def __init__(self, Theta1, Theta2):
        super(NNModel, self).__init__(name='neural_network_model')
        self.dense_1 = tf.keras.layers.Dense(Theta1.shape[0], kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]), activation='sigmoid', input_shape=[X.shape[1]])
        self.dense_2 = tf.keras.layers.Dense(Theta2.shape[0], kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0]), activation='sigmoid')

      def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)

    model = NNModel(Theta1, Theta2)
    pred = model.predict(dataset)

elif option in [2, 3, 4, 5]:

    @tf.function
    def evaluation(Theta1, Theta2, data):
        # inside a @tf.function, I think all variables should be tf types, https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md
        l1 = tf.sigmoid(tf.matmul(data, Theta1[1:,:]) + Theta1[0,:])
        l2 = tf.sigmoid(tf.matmul(l1, Theta2[1:,:]) + Theta2[0,:])
        #l2 = tf.matmul(l1, Theta2[1:,:]) + Theta2[0,:] # One doesn't even need the last sigmoid function because it is monotone increasing and doesn't change the ordering for argmax.
        return l2

    if option==2:
        pred = []
        for entry in dataset:
            #pred.append(evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), entry.numpy().reshape((1,-1)))) # numpy reshape might be faster than tf.reshape
            pred.append(evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), tf.reshape(entry, (1,-1)))) # doing it in TF

        #pred = np.concatenate(pred, axis=0) # this also works
        pred = tf.concat(pred, axis=0)

    elif option==3:

        pred = dataset.map(lambda x: evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), tf.reshape(x, [1,-1])))
        #pred = dataset.map(lambda x: evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), x)) # This doesn't work.
        pred = tf.concat([entry for entry in pred], axis=0)

    elif option==4:
        pred = []
        for batch in dataset:
            for entry in batch:
                pred.append(evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), tf.reshape(entry, (1,-1))))

        pred = tf.concat(pred, axis=0)

    else: # option==5

        pred = dataset.map(lambda x: evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), x))
        #pred = dataset.map(lambda x: evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), tf.reshape(x, [-1,400]))) # This works, in same time.
        pred = tf.concat([entry for entry in pred], axis=0)

else: # option==7

    @tf.function
    def evaluation2(Theta1k, Theta1b, Theta2k, Theta2b, data):
        # inside a @tf.function, I think all variables should be tf types, https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md
        l1 = tf.sigmoid(tf.matmul(data, Theta1k) + Theta1b)
        l2 = tf.sigmoid(tf.matmul(l1, Theta2k) + Theta2b)
        #l2 = tf.matmul(l1, Theta2k) + Theta2b # One doesn't even need the last sigmoid function because it is monotone increasing and doesn't change the ordering for argmax.
        return l2

    pred = dataset.map(lambda x: evaluation2(tf.constant(Theta1[:,1:].T), tf.constant(Theta1[:,0]), tf.constant(Theta2[:,1:].T), tf.constant(Theta2[:,0].T), x))
    #pred = dataset.map(lambda x: evaluation2(tf.constant(Theta1[:,1:].T), tf.constant(Theta1[:,0]), tf.constant(Theta2[:,1:].T), tf.constant(Theta2[:,0].T), tf.reshape(x, [-1,400]))) # This works, in same time.
    pred = tf.concat([entry for entry in pred], axis=0)

    # It is not used in this simplest form:
    #pred = evaluation(tf.constant(Theta1.T), tf.constant(Theta2.T), dataset)

#tf.print(pred)

# The output layer (logits) has 10 units, for digits 1,2,...,9,0. After taking argmax, you have to map the result of argmax, 0,1,2,...,9 to the required 1,2,...,9,0.
pred_digit = tf.map_fn(lambda x: (tf.argmax(x, axis=0, output_type=tf.int32)+1) % 10, pred, dtype=tf.int32)


pred_np = pred_digit.numpy().reshape(-1,1)
print('\nTraining Set Accuracy: {0:.2f}%.'.format(np.mean(pred_np == y) * 100))
print('Expected training error value on complete Training Set (approx.): 97.5%.')

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))
print()

if option in [0, 1, 6]:
    tf.print(model.summary()) # This provides interesting output.

plt.scatter(np.arange(len(y)), y, label='Ground truth')
plt.scatter(np.arange(len(y)), pred_np, marker=".", c='r', label='Prediction')
plt.xlabel('Sample ID')
plt.ylabel('Digit')
plt.legend()
plt.show()

