'''
Backpropagation and training a neural network by TensorFlow 2
https://www.tensorflow.org/tutorials/quickstart/advanced

exec(open('ex4_TF2_custom.py','r').read())

Bence Mélykúti
09-20/03/2018, 04-05/03/2020
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import tensorflow as tf
import matplotlib.pyplot as plt
import time

### User input ###

epochs = 20 # number of epochs in training
logits_option = 1 # {0, 1}, 0 for from_logits=False and tf.sigmoid in Layer 1; 1 for True and no sigmoid
# "Using from_logits=True may be more numerically stable."
deterministic_initialisation = 1 # {0, 1}
regularisation = 0 # {0, 1}
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

def penalty_from_regularisation(model, lambda0, denom):
    pen = 0
    for i in range(2):
        # Both versions for Theta are correct.
        Theta = model.get_weights()[2*i]
        #Theta = model.get_layer(index=i).get_weights()[0]
        pen += lambda0 * 0.5 * np.sum(Theta*Theta, axis=None) / denom
    return pen

start_time = time.time()
lambda0 = 1
y_onehot = tf.one_hot(y.reshape(-1), 10)

dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32),
            y_onehot)).batch(batch_size)
dataset_shuffled = tf.data.Dataset.from_tensor_slices((X.astype(np.float32),
            y_onehot)).shuffle(len(y), reshuffle_each_iteration=True)\
            .batch(batch_size)

class NNModel(tf.keras.Model):

    def __init__(self, Theta1, Theta2):
        super(NNModel, self).__init__(name='neural_network_model')
        self.l0 = tf.keras.layers.Dense(Theta1.shape[0], activation='sigmoid',
                    input_shape=[X.shape[1]])
        self.l1 = tf.keras.layers.Dense(Theta2.shape[0], activation=None)

        if deterministic_initialisation == 1:
            self.l0.kernel_initializer=tf.constant_initializer(Theta1[:,1:].T)
            self.l0.bias_initializer=tf.constant_initializer(Theta1[:,0])
            self.l1.kernel_initializer=tf.constant_initializer(Theta2[:,1:].T)
            self.l1.bias_initializer=tf.constant_initializer(Theta2[:,0])
        else:
            self.l0.kernel_initializer=tf.keras.initializers.he_uniform()
            self.l0.bias_initializer=tf.keras.initializers.he_uniform()
            self.l1.kernel_initializer=tf.keras.initializers.he_uniform()
            self.l1.bias_initializer=tf.keras.initializers.he_uniform()

        if regularisation == 1:
            # Divide lambda0 by 10 to counteract the multiplication of loss by 10 later on.
            self.l0.kernel_regularizer=tf.keras.regularizers.l2(0.1*lambda0 * 0.5 / len(y))
            self.l0.bias_regularizer=None
            self.l1.kernel_regularizer=tf.keras.regularizers.l2(0.1*lambda0 * 0.5 / len(y))
            self.l1.bias_regularizer=None

        if logits_option == 1:
            self.l1.activation = lambda x: tf.gather(x, tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)
        else:
            self.l1.activation = lambda x: tf.gather(tf.sigmoid(x), tf.concat([tf.constant(9, dtype=tf.int32, shape=[1]), tf.range(0,9, dtype=tf.int32)], 0), axis=1)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.l0(inputs)
        return self.l1(x)

model = NNModel(Theta1, Theta2)

if logits_option == 1:
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
else:
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(X, y):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(X)#, training=True)
        loss = loss_object(y, predictions) + tf.reduce_sum(model.losses)
        # + tf.reduce_sum(model.losses) provides the regularisation losses for the two layers, if
        # they are set to other than default None. Cf.
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models#layers_recursively_collect_losses_created_during_the_forward_pass
        # https://stackoverflow.com/questions/56693863/why-does-model-losses-return-regularization-losses
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)

@tf.function
def test_step(X, y):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(X)#, training=False)
    t_loss = loss_object(y, predictions) + tf.reduce_sum(model.losses)

    test_loss(t_loss)
    test_accuracy(y, predictions)

for X, y in dataset:
    test_step(X, y)

# Multiplying by 10 is needed only because the course material divides by number of samples but not by number of classes when taking the mean.
loss_orig = 10*test_loss.result()
accuracy_orig = test_accuracy.result()
if regularisation == 0:
    losswp_orig = loss_orig + penalty_from_regularisation(model, lambda0, len(y))

test_loss.reset_states()
test_accuracy.reset_states()

for epoch in range(epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()

    for X, y in dataset_shuffled:
        train_step(X, y)

    template = 'Start of Epoch {0}:  Loss: {1:.4f}, accuracy: {2:.2f}%.'
    print(template.format(epoch+1, 10*train_loss.result(), 100*train_accuracy.result()))

for X, y in dataset:
    test_step(X, y)

loss = 10*test_loss.result()
accuracy = test_accuracy.result()
template = 'End  of  Epoch {0}:  Loss: {1:.4f}, accuracy: {2:.2f}%.'
print(template.format(epoch+1, loss, 100*accuracy))

if regularisation == 0:
    losswp = loss + penalty_from_regularisation(model, lambda0, len(y))

if regularisation == 0:
    if deterministic_initialisation == 1:
        print('\nExpected loss at parameters loaded from ex4weights.mat (approx.): 0.287629.')
    else:
        print()
    print('Complete Training Set loss before the start of training: {0:.6f}.'.format(loss_orig))
    print('Complete Training Set loss after training: {0:.6f}.'.format(loss))

if deterministic_initialisation == 1:
    print('\nExpected regularised loss at parameters loaded from ex4weights.mat (approx.):\n0.383770.')
else:
    print()
if regularisation == 0:
    print('Manually computed regularised loss before the start of training: {0:.6f}.'.format(losswp_orig))
    print('Manually computed regularised loss after training: {0:.6f}.'.format(losswp))
else:
    print('Regularised loss before the start of training: {0:.6f}.'.format(loss_orig))
    print('Regularised loss after training: {0:.6f}.'.format(loss))


if deterministic_initialisation == 1:
    print('\nExpected original training accuracy on complete Training Set (approx.): 97.5%.')
else:
    print()
print('Training Set Accuracy before the start of training: {0:.2f}%.'.format(accuracy_orig * 100))
print('Training Set Accuracy after training: {0:.2f}%.'.format(accuracy * 100))

print('\nTime elapsed: {:.2f} sec.'.format(time.time() - start_time))

