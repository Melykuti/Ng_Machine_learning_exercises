'''
One-versus-all regularised logistic regression

Bence Mélykúti
23/02/2018
'''

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import scipy.io # to open Matlab's .mat files
import time

# own implementation, not really recommended or necessary:
def reglogreg_manual(X, y, num_labels, lambda0):
#    all_theta = np.zeros((num_labels, 1+X.shape[1]))
    data_prediction = np.zeros((X.shape[0], num_labels))

    for i in range(num_labels):
        lreg = linear_model.LogisticRegression(C=1/lambda0, fit_intercept=True, intercept_scaling=100)
        y_i = np.zeros_like(y)
        y_i[y==i] = 1
        y_i[y!=i] = 0

        lreg.fit(X, y_i) # there is no first column with all 1s, but it is taken care of by fit_intercept=True
#        all_theta[i, 0] = lreg.intercept_
#        all_theta[i, 1:] = np.ravel(lreg.coef_)

        data_prediction[:,i] = np.ravel(lreg.predict_proba(X)[:,1])

    pred = np.argmax(data_prediction, axis=1)
    return pred, data_prediction#, all_theta

# built-in version; benchmarking shows that it's not much faster
def reglogreg(X, y, lambda0):
    lreg = linear_model.LogisticRegression(C=1/lambda0, fit_intercept=True, intercept_scaling=100, multi_class='ovr') # ovr = one-vs-rest
    lreg.fit(X,y)
    pred = lreg.predict(X)
    data_prediction = lreg.predict_proba(X)
    trainacc = lreg.score(X,y)
    return pred, data_prediction, trainacc

mat = scipy.io.loadmat('../machine-learning-ex3/ex3/ex3data1.mat')
mat['y'][mat['y'][:]==10]=0 # this sets character identifier 10 to 0
#mat['y'] = mat['y'] % 10 # this is another way of setting 10 to 0
num_labels=10 # number of classes

X = mat['X']
y = mat['y']


# Although PolynomialFeatures could cope with it, we avoid using polynomial features as we have many (400), not only 2 variates in each sample.
#degree=6
#X=mapFeature(X[:,0], X[:,1], degree)


# Regularised logistic regression

# regularisation parameter
lambda0 = 1 # 1e-2 is a better value; set to tiny value (e.g. 1e-10) to achieve no regularisation

start_time = time.time()

# Option 1
#pred, data_prediction = reglogreg_manual(X, y, num_labels, lambda0) # takes 3.04 sec with lambda0=1, 8 sec with lambda0=1e-2

# Option 2
# https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists#843293
if 'trainacc' in globals():
    del trainacc
pred, data_prediction, trainacc = reglogreg(X, y, lambda0) # takes 2.99 sec with lambda0=1, 7.7 sec with lambda0=1e-2
print('Time elapsed: {:.2f} sec'.format(time.time() - start_time))

print('{0} data points out of {1} correctly classified.'.format(np.sum(np.ravel(y)==pred),X.shape[0]))
print('Train Accuracy:')

if 'trainacc' in globals():
    print(trainacc * 100, 'percent')
#else: # This is an alternative calculation.
print(np.mean(np.ravel(y)==pred) * 100, 'percent')
print('Expected accuracy (with lambda = 1): 94.9 (approx)')

