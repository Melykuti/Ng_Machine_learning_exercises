'''
Backpropagation and training a neural network by a TensorFlow custom estimator

-How are the network parameters in the optimisation initialised: with random values?
ex3_nn.py shows you how to initialise them any way you like.

-Can I add regularisation?
I haven't found out yet how to include the network weights in the loss function in order to achieve regularisation.

Bence Mélykúti
09-26/03/2018
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

'''
# The network parameters are here only for info, they are not actually used.
input_layer_size == 400     # 20x20 Input Images of Digits
hidden_layer_size == 25     # 25 hidden units
num_labels == 10            # 10 labels, from 1 to 10   
                            # (note that we have mapped "0" to label 10)
'''

'''
def train_input_fn(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #.batch(X.shape[0]) #.batch(50) #.batch(X.shape[0]-1)

    (x, ylabels) = dataset.make_one_shot_iterator().get_next() # https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
# https://www.tensorflow.org/programmers_guide/datasets#consuming_values_from_an_iterator

    return {'pixels': x}, tf.cast(ylabels, dtype=tf.int32)
#    (x, ylabels) = dataset.make_one_shot_iterator().get_next() # https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
# https://www.tensorflow.org/programmers_guide/datasets#consuming_values_from_an_iterator
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

# Compulsory arguments of model: https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#__init__
#def model(X, y, mode, params): # doesn't work
#def model(features, y, mode, params): # doesn't work
def model(features, labels, mode, params):
    #n_features = len(params['feature_columns'])

    # net = tf.feature_column.input_layer(X, params['feature_columns']) # doesn't work
    net = tf.feature_column.input_layer(features, params['feature_columns'])


    ## Option 1 ##

    # This is used in ex4_premade.py because that is what tf.estimator.DNNClassifier uses:
    net1 = tf.layers.dense(net, units=params['hidden_units'][0], use_bias=True, activation=tf.nn.relu, name='hiddenlayer')
    # This is used originally in the exercises and in ex4.py:
    #net1 = tf.layers.dense(net, units=params['hidden_units'][0], use_bias=True, activation=tf.nn.sigmoid, name='hiddenlayer')
    ##net1 = tf.layers.dense(tf.reshape(x, [-1, n_features]), units=params['hidden_units'][0], use_bias=True, activation=tf.nn.sigmoid) # the tf.reshape is needed if batch size = 1 (there is no .batch())

    # activation=tf.nn.sigmoid would be wrong because the loss function turns the logits values into sigmoid(logits), so there must be no application of sigmoid in the output of this layer:
    logits = tf.layers.dense(net1, units=params['n_classes'], use_bias=True, activation=None, name='logitslayer')

    ## End of Option 1 ##

    ## Option 2 ##
    '''
    # ================ Part 2: Loading Pameters ================
    # Here we load some pre-initialized neural network parameters.
    # Their sizes override params['hidden_units'] and params['n_classes']
    params = scipy.io.loadmat('../machine-learning-ex4/ex4/ex4weights.mat')
    Theta1 = params['Theta1']   # Theta1 has size 25 x 401
    Theta2 = params['Theta2']   # Theta2 has size 10 x 26

    net1 = tf.layers.dense(net, units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0]), name='hiddenlayer')
    ##net1 = tf.layers.dense(tf.reshape(x, [-1, n_features]), units=Theta1.shape[0], use_bias=True, activation=tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(Theta1[:,1:].T), bias_initializer=tf.constant_initializer(Theta1[:,0])) # the tf.reshape is needed if batch size = 1 (there is no .batch())

    # activation=tf.nn.sigmoid would be wrong because the loss function turns the logits values into sigmoid(logits), so there must be no application of sigmoid in the output of this layer:
    logits = tf.layers.dense(net1, units=Theta2.shape[0], use_bias=True, activation=None, kernel_initializer=tf.constant_initializer(Theta2[:,1:].T), bias_initializer=tf.constant_initializer(Theta2[:,0]), name='logitslayer')
    '''
    ## End of Option 2 ##


    # Rearranging the columns: digits [1,2,...,9,0] are mapped to digits [0,1,2,...,9], that is, column 9 (digit 0, encoded with position 10 of [1,10]) must come first, the rest must be shifted up by one.
    logits = tf.gather(logits, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)],0), axis=1)
    #logits = tf.gather(logits, np.concatenate(([9], np.arange(0,9))), axis=1) # equivalent, with np arrays


    ## Option a: One-hot ##
    
    # Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
    y_idcc = tf.feature_column.categorical_column_with_identity(key='labels', num_buckets=10)
    y_onehot = tf.feature_column.indicator_column(y_idcc)
    #y_layer = tf.feature_column.input_layer({'labels': y}, y_onehot) # doesn't work
    y_layer = tf.feature_column.input_layer({'labels': labels}, y_onehot)

    # This is used in ex4_premade.py because that is what tf.estimator.DNNClassifier uses:
    loss = tf.losses.softmax_cross_entropy(y_layer, logits) * 10
    # This is used originally in the exercises and in ex4.py:
    #loss = tf.losses.sigmoid_cross_entropy(y_layer, logits) * 10
    
    ## End of Option a: One-hot ##

    ## Option b: Single column with ordinal class representation ##
    '''
    # Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
    y_numc = tf.feature_column.numeric_column(key='labels', dtype=tf.int32)
    #y_layer = tf.feature_column.input_layer({'labels': y}, y_numc) # doesn't work
    #print(labels.dtype)
    y_layer = tf.feature_column.input_layer({'labels': labels}, y_numc)

    # This is used in ex4_premade.py because that is what tf.estimator.DNNClassifier uses:
    y_layer = tf.cast(y_layer, dtype=tf.int32) # tf.feature_column.input_layer returns float32
    loss = tf.losses.sparse_softmax_cross_entropy(y_layer, logits) * 10\
    # tf.nn.sigmoid_cross_entropy_with_logits doesn't work with more than 2 classes,
    # unless you manually apply it in a one-vs-all fashion for all classes.
    '''
    ## End of Option b: Single column with ordinal class representation ##

    predicted_digits = tf.map_fn(lambda x: tf.argmax(x, axis=0), logits, dtype=tf.int64) # if columns are rearranged by tf.gather, digits are stored as [0,1,2,...,9]
    ## In custom_estimator.py: predicted_classes = tf.argmax(logits, 1) -- why 1? Probably because the batch elements are down the zeroth dimension.

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_digits[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute evaluation metrics.
    #accuracy = tf.metrics.accuracy(labels=y, # doesn't work
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_digits,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(X, y, batch_size):

    # Feature columns
    feature_columns = [tf.feature_column.numeric_column("pixels", shape=[400], dtype=tf.float32)]

    '''
    # Neural network in ex4_premade.py
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # One hidden layer of 25 nodes.
        hidden_units=[25],
        # The model must choose between 10 classes.
        n_classes=10,
        activation_fn=tf.nn.sigmoid)
    '''

    classifier = tf.estimator.Estimator(
        #model_fn=lambda features,y,mode,params: model(features, y, mode, params), # doesn't work: ValueError: model_fn (<function main.<locals>.<lambda> at 0x7f57502a6c80>) must include features argument.
        model_fn=model,
        params={
            'feature_columns': feature_columns,
            # One hidden layer of 25 nodes.
            'hidden_units': [25],
            # The model must choose between 10 classes.
            'n_classes': 10,
            'batch_size': batch_size})

    # Training the model
    classifier.train(input_fn=lambda:train_input_fn(X, y, batch_size), steps=500)
    # Doesn't work without lambda:
    #classifier.train(input_fn=train_input_fn(X, y, batch_size), steps=1)

    '''
    # Evaluating the model
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(X, y, batch_size))#, steps=10) # steps is allowed here
    # Doesn't work without lambda:
    #eval_result = classifier.evaluate(input_fn=eval_input_fn(X, y, batch_size))
    print('Training set accuracy: {accuracy:0.3f}'.format(**eval_result))
    '''
    return classifier


X, y = load_data()
batch_size=X.shape[0]

start_time = time.time()

tf.reset_default_graph() # not strictly necessary

'''
# This runs on its own:
classifier = main(X, y, batch_size)
classifier.evaluate(input_fn=lambda:eval_input_fn(X, y, batch_size))
'''

#tf.app.run(main(X, y, batch_size)) # doesn't work on its own

with tf.Session() as sess:
    #classifier = sess.run(main(X, y, batch_size)) # this doesn't work
    classifier = main(X, y, batch_size)

    # Evaluating the model
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(X, y, batch_size))#, steps=10) # steps is allowed here
    # Doesn't work without lambda:
    #eval_result = classifier.evaluate(input_fn=eval_input_fn(X, y, batch_size))
    print('Training set accuracy: {accuracy:0.3f}'.format(**eval_result))


print('\nTime elapsed: {:.2f} sec'.format(time.time() - start_time))
