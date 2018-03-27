'''
Backpropagation and training a neural network by a TensorFlow pre-made estimator

-Can I initialise the optimisation any way I like? Probably only in a custom estimator, the way we did it in ex3_nn.py.
How are the network parameters in the optimisation initialised: with random values?

-Can I change the loss function?
https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
"Loss is calculated by using softmax cross entropy."
In custom estimators, the loss function can be customised.

-Can I extract the edge weights from the neural network?
Yes, but I can only do it once the training has completed. Try classifier.get_variable_names(), and from the variables, to retrieve e.g. 'dnn/hiddenlayer_0/bias':
classifier.get_variable_value('dnn/hiddenlayer_0/bias')

https://stackoverflow.com/questions/36193553/get-the-value-of-some-weights-in-a-model-trained-by-tensorflow

-Can I add regularisation?
There is the possibility of some kind of regularisation in the optimiser; https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor
optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    )
or in the training routine: https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier
tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001,
    l2_regularization_strength=0.001)
It's not clear what they do without delving deeper into the subject.

In conclusion, to regularise network parameters, define an appropriate loss function in a custom estimator.


Bence Mélykúti
09-17/03/2018
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

'''
Exercise uses these parameters:
input_layer_size == 400     # 20x20 Input Images of Digits
hidden_layer_size == 25     # 25 hidden units
num_labels == 10            # 10 labels, from 1 to 10   
                            # (note that we have mapped "0" to label 10)
'''


def load_data():
    data = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4data1.mat')
    X = data['X']#.astype(np.float32)
    y = data['y']
    y = y % 10
    return X, y


'''
In the train_input_fn() and eval_input_fn(), there is no .make_one_shot_iterator().get_next() in the official example:
https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
or in the Getting Started with TensorFlow tutorial:
https://www.tensorflow.org/get_started/premade_estimators#create_input_functions
But it is included in the Creating Custom Estimators tutorial:
https://www.tensorflow.org/get_started/custom_estimators#write_an_input_function
It seems to run both with and without it.
'''
def train_input_fn(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(({'pixels': X}, y.astype(np.int32)))
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.repeat().batch(batch_size)
    return dataset#.make_one_shot_iterator().get_next()

def eval_input_fn(X, y, batch_size):
#    dataset = tf.data.Dataset.from_tensor_slices((X,y)).batch(batch_size)
#    (x, ylabels) = dataset.make_one_shot_iterator().get_next()
#    return {'pixels': x}, tf.cast(ylabels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(({'pixels': X}, y.astype(np.int32)))
    dataset = dataset.batch(batch_size)
    return dataset

def model(X, y, batch_size):

    # Feature columns
    feature_columns = [tf.feature_column.numeric_column("pixels", shape=[400], dtype=tf.float32)]

    '''
We don't use one-hot encoding like we did in ex4.py, see
https://stackoverflow.com/questions/47120637/typeerror-unsupported-callable-using-dataset-with-estimator-input-fn
y_idcc = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
y_onehot = tf.feature_column.indicator_column(y_idcc)
    '''

    # Neural network
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # One hidden layer of 25 nodes.
        hidden_units=[25],
        # The model must choose between 10 classes.
        n_classes=10,
        optimizer='Adagrad',
        activation_fn=tf.nn.sigmoid)

    # Training the model
    classifier.train(input_fn=lambda:train_input_fn(X, y, batch_size), steps=500)
    # Doesn't work without lambda:
    #classifier.train(input_fn=train_input_fn(X, y, batch_size), steps=1)

    return classifier


X, y = load_data()
batch_size=X.shape[0]

start_time = time.time()

tf.reset_default_graph() # not strictly necessary

classifier = model(X, y, batch_size)


# Evaluating the model
eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(X, y, batch_size))
# Doesn't work without lambda:
#eval_result = classifier.evaluate(input_fn=eval_input_fn(X, y, batch_size))
print('Training set accuracy: {accuracy:0.3f}'.format(**eval_result))

print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))

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
# These can now be passed into ex4.py, just comment out the definitions of Theta1 and Theta2 in ex4.py to avoid overwriting them.

'''
print('\nComparison of penalties for the supplied network parameters and\n those trained right now:')
lambda0 = 1
print(lambda0 * 0.5 * (np.sum(params['Theta1'][:,1:]*params['Theta1'][:,1:], axis=None) + np.sum(params['Theta2'][:,1:]*params['Theta2'][:,1:], axis=None)) / batch_size)
print(lambda0 * 0.5 * (np.sum(Theta1[:,1:]*Theta1[:,1:], axis=None) + np.sum(Theta2[:,1:]*Theta2[:,1:], axis=None)) / batch_size)
The penalty is
0.0961 in the supplied network,
0.2877 in the current training.

#The loss and regularised loss are 3-4 times higher with my training than with the supplied network parameters. One can extract the logits values using ex3_nn.py (define ll in addition to dd). Then the value of np.sum(np.abs(ll)):
5023.234776334117 for the supplied network,
21239.714247487827 for current training.
'''

