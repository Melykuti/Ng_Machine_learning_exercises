'''
Backpropagation and training a neural network by a TensorFlow pre-made estimator
https://www.tensorflow.org/guide/estimator
https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier

f=open('ex4_TF2_estimator.py','r'); exec(f.read()); f.close()


-Can I initialise the optimisation any way I like? tf.estimator.DNNClassifier has an argument, warm_start_from, to specify a model to load from disk.

-How are the network parameters in the optimisation initialised: with random values?

-Can I change the loss function?
It seems not.
https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#example_2
"Loss is calculated by using softmax cross entropy."

-Can I extract the edge weights from the neural network?
Yes, but I can only do it once the training has completed. Try classifier.get_variable_names(), and from the variables, to retrieve e.g. 'dnn/hiddenlayer_0/bias':
classifier.get_variable_value('dnn/hiddenlayer_0/bias')
https://stackoverflow.com/questions/36193553/get-the-value-of-some-weights-in-a-model-trained-by-tensorflow

-Can I add regularisation?
It seems not.

Bence Mélykúti
09-17/03/2018, 04/03/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

### User input ###

steps = 100 # number of steps in training
data_pipeline_option = 0 # {0, 1}
batch_size = 5000 # try: 50 or 3900 or 4999 or 5000 (which is X.shape[0]) or 50000

### End of input ###

# The network parameters are here for info, they are not actually used.
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

def load_data():
    data = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4data1.mat')
    X = data['X']
    y = data['y']
    y = y % 10 # Transforming 10 to 0, which is its original meaning.
    return X, y

'''
We don't use one-hot encoding like we did in ex4_TF2_Keras.py, see
https://stackoverflow.com/questions/47120637/typeerror-unsupported-callable-using-dataset-with-estimator-input-fn
'''
def y_into_onehot(y):
    # Turning y into one hot encoding.
    return tf.one_hot(y.reshape(-1), 10)

if data_pipeline_option == 0:
    def input_fn_train():
        X, y = load_data()
        #y_onehot = y_into_onehot(y)
        dataset_shuffled = tf.data.Dataset.from_tensor_slices(({'pixels': X.astype(np.float32)},
                y.astype(np.int32)))\
                .shuffle(len(y), reshuffle_each_iteration=True)\
                .repeat()\
                .batch(batch_size)
        return dataset_shuffled

    def input_fn_eval():
        X, y = load_data()
        #y_onehot = y_into_onehot(y)
        dataset = tf.data.Dataset.from_tensor_slices(({'pixels': X.astype(np.float32)},
                y.astype(np.int32)))\
                .batch(len(y))
        return dataset
else:
    def input_fn_train(X, y, batch_size):
        #y_onehot = y_into_onehot(y)
        dataset_shuffled = tf.data.Dataset.from_tensor_slices(({'pixels': X.astype(np.float32)},
                y.astype(np.int32)))\
                .shuffle(len(y), reshuffle_each_iteration=True)\
                .repeat()\
                .batch(batch_size)
        return dataset_shuffled

    def input_fn_eval(X, y, batch_size):
        #y_onehot = y_into_onehot(y)
        dataset = tf.data.Dataset.from_tensor_slices(({'pixels': X.astype(np.float32)},
                y.astype(np.int32)))\
                .batch(len(y))
        return dataset


def model(batch_size, X=None, y=None):

    # Feature columns
    feature_columns = [tf.feature_column.numeric_column("pixels", shape=[400], dtype=tf.float32)]

    # Neural network
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # One hidden layer of 25 nodes.
        hidden_units=[25],
        # The model must choose between 10 classes.
        n_classes=10,
        optimizer='Adam',
        activation_fn=tf.nn.sigmoid)

    # Training the model
    if data_pipeline_option == 0:
        classifier.train(input_fn=input_fn_train, steps=steps)
        # And not input_fn=input_fn_train().
    else:
        classifier.train(input_fn=lambda: input_fn_train(X, y, batch_size), steps=steps)
        # Doesn't work without lambda:
        #classifier.train(input_fn=input_fn_train(X, y, batch_size), steps=1)#50)

    return classifier

start_time = time.time()

if data_pipeline_option == 0:
    classifier = model(batch_size)
    eval_result = classifier.evaluate(input_fn=input_fn_eval)
else:
    X, y = load_data()
    classifier = model(batch_size, X, y)
    eval_result = classifier.evaluate(input_fn=lambda: input_fn_eval(X, y, batch_size))

print('\nTraining set accuracy: {accuracy:0.3f}.'.format(**eval_result))

print('\nTime elapsed: {:.2f} sec.'.format(time.time() - start_time))

# Extracting neural network parameters to store in Theta1, Theta2 matrices
# classifier.get_variable_names() tells you which variables are available.
Theta1=np.zeros((25,401))
Theta1[:,0] = classifier.get_variable_value('dnn/hiddenlayer_0/bias')
Theta1[:,1:] = classifier.get_variable_value('dnn/hiddenlayer_0/kernel').T
Theta2=np.zeros((10,26))
Theta2[:,0] = classifier.get_variable_value('dnn/logits/bias')
Theta2[:,1:] = classifier.get_variable_value('dnn/logits/kernel').T
# This was mapped by classifier to [0,1,2,...,9] via Theta2 because y had values in {0,1,2,...,9}. We map it to [1,2,...,9,0]:
Theta2 = np.concatenate((Theta2[1:,:], Theta2[np.newaxis,0,:]), axis=0)
# These can now be passed into ex4_TF2.py, just comment out the definitions of Theta1 and Theta2 in ex4.py to avoid overwriting them.

'''
print('\nComparison of penalties for the supplied network parameters and\n those trained right now:')
lambda0 = 1
print(lambda0 * 0.5 * (np.sum(params['Theta1'][:,1:]*params['Theta1'][:,1:], axis=None) + np.sum(params['Theta2'][:,1:]*params['Theta2'][:,1:], axis=None)) / batch_size)
print(lambda0 * 0.5 * (np.sum(Theta1[:,1:]*Theta1[:,1:], axis=None) + np.sum(Theta2[:,1:]*Theta2[:,1:], axis=None)) / batch_size)
The penalty is
0.0961 in the supplied network,
~0.012 in the current training.
'''
