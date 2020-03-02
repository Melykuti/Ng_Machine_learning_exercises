# Python solutions for Andrew Ng's Machine Learning course on Coursera (`scikit-learn` and `TensorFlow`)


[Andrew Ng's online _Machine Learning_ course on Coursera](https://www.coursera.org/learn/machine-learning) has become a bit of a cultural phenomenon like _Harry Potter_ or _Star Wars_: everybody seems to have heard of it, watched it, or completed it. That is, if you move in the right circles.

The course includes computer programming exercises, which are, for didactic reasons, in `octave`/`Matlab`. In order to explain the methods thoroughly, some exercises deal with implementation details that are useful to understand once but not very relevant in everyday applications (e.g. feedforward propagation and backpropagation for neural networks).

My goal with this repository is to solve some of the exercises in `Python 3`, which is the most popular language in contemporary data science, using standard packages that one would use in real life to create reliable and fast code. I use `TensorFlow` for the neural networks (I developed my code in version 1.5) and the versatile `scikit-learn` package for the rest. File names ending in `_TF2` are adapted to the significantly different `TensorFlow 2.0`.


#### Author

With questions or comments, please contact [Bence Mélykúti](https://github.com/Melykuti). Follow him on [Twitter](https://twitter.com/BMelykuti).


#### Exercises

* **ex1.py** Linear regression with one variable, ridge regression (L2 regularisation), linear regression with stochastic gradient descent. Plotting.
* **ex1\_multi.py** Linear regression with multiple variables. Feature normalisation.
* **ex2.py** Sigmoid function. Logistic regression. Plotting a linear decision boundary.
* **ex2\_reg.py** Regularised logistic regression. Generating polynomial features. Plotting the decision boundary.
* **ex3.py** One-versus-all regularised logistic regression
* **ex3_nn.py** Neural networks. Forward propagation in an already trained network in `TensorFlow` (to use the network for classification).
* **ex3_nn_TF1inTF2.py** Neural networks. Forward propagation in an already trained network using `TensorFlow 1` code of `ex3_nn.py` in compatibility mode in `TensorFlow 2` (to use the network for classification).
* **ex3_nn_TF2.py** Neural networks. Forward propagation in an already trained network in `TensorFlow 2.0` (to use the network for classification).
* **ex4.py** Neural networks. Forward propagation in an already trained network in `TensorFlow`. Computing the regularised cost function.
* **ex4_TF1inTF2.py** Neural networks. Forward propagation in an already trained network using `TensorFlow 1` code of `ex4.py` in compatibility mode in `TensorFlow 2`. Computing the regularised cost function.
* **ex4_TF2.py** Neural networks. Forward propagation in an already trained network in `TensorFlow 2.0`. Computing the regularised cost function.
* **ex4_TF2_evaluate.py** Similar to `ex4_TF2.py` but uses the `tf.keras.Model.evaluate()` method.
* **ex4\_premade.py** Backpropagation and training a neural network with a `TensorFlow` pre-made estimator
* **ex4\_custom.py** Backpropagation and training a neural network with a `TensorFlow` custom estimator
* **ex5.py** Regularised linear regression, i.e. ridge regression. Training set, cross-validation set, test set. Bias‒variance trade-off. Learning curves. Regularised polynomial regression. Selecting the regularisation parameter using a cross-validation set.
* **ex6.py** Support vector machines. Plotting the decision boundary. Gaussian kernel (radial basis function kernel). Selecting the regularisation parameter using a cross-validation set.

#### Dependencies and setup

You should retrieve the exercise sheets and data files [from the course website](https://www.coursera.org/learn/machine-learning). If your exercise code is stored in folders of the form  
`path/to/exercises/machine-learning-ex?/ex?`,  
then copy these python files to this level for the data input to work:  
`path/to/exercises/python`.

#### License

The software in the repository _Melykuti/Ng\_Machine\_learning\_exercises_ is licensed under the [3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).

Copyright (c) 2018-2020, Bence Mélykúti
