'''
Regularised linear regression, i.e. ridge regression. Training set, cross-validation set, test set. Bias‒variance trade-off. Learning curves. Regularised polynomial regression. Selecting the regularisation parameter using a cross-validation set.

Bence Mélykúti
24-25/02/2018
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import PolynomialFeatures
import random

def linearRegCostFunction(X, y, theta, lambda0):
    return np.sum(np.square(np.subtract(theta[0] + np.dot(X, np.asarray(theta[1:])), np.ravel(y))))/(2*X.shape[0]) + lambda0/(2*X.shape[0])*np.sum(np.square(np.asarray(theta[1:])))

def learningCurve(X, y, Xval, yval, lambda0, iterations):
    error_train=[]
    error_val=[]
    for i in range(X.shape[0]):
        ridge = linear_model.Ridge(alpha=lambda0, fit_intercept=True, max_iter=iterations, tol=1e-10)
        ridge.fit(X[:i+1,],y[:i+1])
#        theta = np.concatenate((np.asarray(ridge.intercept_).reshape(1), ridge.coef_[0]))
        theta = np.concatenate((ridge.intercept_, ridge.coef_[0]))
        #print(theta)
        error_train = error_train + [linearRegCostFunction(X[:i+1,], y[:i+1,], theta, 0)]
        error_val = error_val + [linearRegCostFunction(Xval, yval, theta, 0)]
    return error_train, error_val

# an advanced variant of learningCurve where training and CV sets are sampled from the respective full sets sample_size times, and error_train and error_val result from averaging
def learningCurve_w_averaging(X, y, Xval, yval, lambda0, iterations, sample_size):
    # random data selection with or without replacement:
    with_replacement=False # if X.shape[0] > Xval.shape[0], then it must be True
    # It's probably best to set with_replacement=True and for i in range(max(X.shape[0], Xval.shape[0])): ...

    error_train=np.zeros((sample_size, X.shape[0]))
    error_val = np.zeros((sample_size, X.shape[0]))
    ridge = linear_model.Ridge(alpha=lambda0, fit_intercept=True, max_iter=iterations, tol=1e-10)
    for i in range(X.shape[0]):
        for l in range(sample_size):
            if with_replacement==1:
                indxtr=random.choices(range(X.shape[0]), k=i+1)
                indxcv=random.choices(range(Xval.shape[0]), k=i+1)
            else: # without_replacement:
                indxtr=random.sample(range(X.shape[0]), k=i+1)
                indxcv=random.sample(range(Xval.shape[0]), k=i+1)

            ridge.fit(X[indxtr,],y[indxtr])
            theta = np.concatenate((ridge.intercept_, ridge.coef_[0]))
            error_train[l,i] = linearRegCostFunction(X[indxtr,], y[indxtr,], theta, 0)
            error_val[l,i] = linearRegCostFunction(Xval[indxcv,], yval[indxcv,], theta, 0)
    #print(error_train.shape)
    error_train=np.mean(error_train,axis=0)
    error_val = np.mean(error_val,axis=0)
    #print(error_train.shape)
    #print(sample_size)
    return error_train, error_val

def validationCurve(X_poly, y, X_poly_val, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train=[]
    error_val=[]
    for i in range(len(lambda_vec)):
        ridge = linear_model.Ridge(alpha=lambda_vec[i], fit_intercept=True, max_iter=iterations, tol=1e-10)
        ridge.fit(X_poly,y)
#        theta = np.concatenate((np.asarray(ridge.intercept_).reshape(1), ridge.coef_[0]))
        theta = np.concatenate((ridge.intercept_, ridge.coef_[0]))
        error_train = error_train + [linearRegCostFunction(X_poly, y, theta, 0)]
        error_val = error_val + [linearRegCostFunction(X_poly_val, yval, theta, 0)]
    return lambda_vec, error_train, error_val

# http://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features
# cf. mapFeature(X1, X2, degree) in ex2_reg.py
def polyFeatures(X, degree): # X must be 1-dim vector, >=2-dim matrices not allowed
# The output is X[i,:] = np.array([X[i],X[i]^2,...,X[i]^degree])
    X=np.asarray(X)
    X=X.reshape((X.shape[0],1))
    poly = PolynomialFeatures(degree) # The first column of output is all 1.
    #print(poly.fit_transform(X).shape)
    #print(np.delete(poly.fit_transform(X),0,1).shape)
    return np.delete(poly.fit_transform(X),0,1) # Remove first column with all 1.

# =========== Part 1: Loading and Visualising Data =============
data = scipy.io.loadmat('../machine-learning-ex5/ex5/ex5data1.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']


# Visualisation

plt.scatter(X, y, marker="x", c='r', label='Train')
plt.scatter(Xval, yval, marker="x", c='0.85', label='Cross Validation')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.legend()
plt.show()


# ======= Part 2: Linear regression with regularisation, i.e. ridge regression

# evaluating cost function
lambda0=1 # "Regularization strength; must be a positive float. Larger values specify stronger regularization."

theta=[1, 1]
print('With theta = ', theta, '\nCost computed =', linearRegCostFunction(X, y, theta, lambda0))
# np.sum(np.square(theta[0]+theta[1]*X[:,0]-np.ravel(y)))/(2*X.shape[0]) +
# lambda0/(2*X.shape[0])*np.sum(np.square(np.asarray(theta[1:]))))
print('Expected cost value (approx) 303.993192')

# =========== Part 4: Linear regression, ridge regression

lambda0 = 0 # "Regularization strength; must be a positive float. Larger values specify stronger regularization."
iterations = 1500
ridge = linear_model.Ridge(alpha=lambda0, fit_intercept=True, max_iter=iterations, tol=1e-5)
ridge.fit(X,y)
print('\nTheta found by ridge regression:')
print(ridge.intercept_, ridge.coef_)

data_prediction = ridge.predict(X)

# Visualisation

plt.scatter(X, y, marker="x", c='r')
plt.plot(X, data_prediction, color='blue', linewidth=2)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.legend()
plt.show()


# =========== Part 5: Learning curves


error_train, error_val = learningCurve(X, y, Xval, yval, lambda0, iterations)

# Only for testing as an alternative to Xval, yval, not meant to be used; here 5th & 9th points are almost level, in Xval, there is a greater drop:
#error_train, error_val = learningCurve(X, y, Xtest, ytest, lambda0, iterations)

plt.plot(1+np.arange(X.shape[0]), error_train, 'b-', label='Train')
plt.plot(1+np.arange(X.shape[0]), error_val, 'g-', label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.show()


# =========== Part 6: Feature Mapping for Polynomial Regression

degree=8
X_poly = polyFeatures(X, degree)
#print('X_poly.shape',X_poly.shape)
#print(X_poly[0:3,])

# feature normalisation with sklearn.preprocessing.StandardScaler (cf. ex1_multi.py):
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = preprocessing.StandardScaler().fit(X_poly)
X_poly = scaler.transform(X_poly) # scaled X
#print('X_poly.shape',X_poly.shape)
#print(X_poly[0:3,])
#mu = scaler.mean_
#sigma = scaler.var_

X_poly_val = polyFeatures(Xval, degree)
X_poly_val = scaler.transform(X_poly_val)

X_poly_test = polyFeatures(Xtest, degree)
X_poly_test = scaler.transform(X_poly_test)


# =========== Part 7: Learning Curve for Polynomial Regression

lambda0=3 # "Regularization strength; must be a positive float. Larger values specify stronger regularization."
iterations = 15000
ridge = linear_model.Ridge(alpha=lambda0, fit_intercept=True, max_iter=iterations, tol=1e-8)
ridge.fit(X_poly,y)
print('\nTheta found by polynomial ridge regression:')
print(ridge.intercept_, ridge.coef_)

fig1 = plt.figure()
plt.scatter(X, y, marker="x", c='r')
xx=np.arange(np.min(X)-15, np.max(X)+7, 0.05)
xx_poly=polyFeatures(xx, degree)
xx_poly=scaler.transform(xx_poly)
plt.plot(xx,ridge.predict(xx_poly),'--')
fig1.suptitle('Polynomial Regression Fit (lambda = {:f})'.format(lambda0))
# This is clearly different from Fig. 4 of the exercise sheet -- they seem to be separate local minima because both are good fits to training data.
#plt.show()

fig2 = plt.figure()
ax = plt.axes()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda0, iterations)
plt.plot(1+np.arange(X.shape[0]), error_train, 'b-', label='Train')
plt.plot(1+np.arange(X.shape[0]), error_val, 'g-', label='Cross Validation')
fig2.suptitle('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda0))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
ax.set_xlim((0,13))
ax.set_ylim((0,100))
plt.legend()
plt.show()

print('\n\nPolynomial ridge regression (lambda = {:f})\n'.format(lambda0))
print('# Training examples\tTrain error\tCross validation error')
for i in range(X.shape[0]):
    print('  \t{0:d}\t\t{1:f}\t{2:f}'.format(i+1, error_train[i], error_val[i]))


# =========== Part 8: Validation for Selecting Lambda

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

print('\nlambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' {:f}\t{:f}\t{:f}'.format(lambda_vec[i], error_train[i], error_val[i]))

plt.plot(lambda_vec, error_train, 'b-', label='Train')
plt.plot(lambda_vec, error_val, 'g-', label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
ax = plt.axes()
ax.set_ylim((0,20))
plt.legend()
plt.show()

# 3.4 Optional (ungraded) exercise: Computing test set error

# argmin, argmax in Python: "argmin(error_val)" == error_val.index(min(error_val))
# https://scaron.info/blog/argmax-in-python.html
# https://lemire.me/blog/2008/12/17/fast-argmax-in-python/
lambda_best=lambda_vec[error_val.index(min(error_val))]
ridge = linear_model.Ridge(alpha=lambda_best, fit_intercept=True, max_iter=iterations, tol=1e-8)
ridge.fit(X_poly,y)
#theta = np.concatenate((np.asarray(ridge.intercept_).reshape(1), ridge.coef_[0]))
theta = np.concatenate((ridge.intercept_, ridge.coef_[0]))
print('\nTest error with optimal lambda (lambda = {:f}): '.format(lambda_best), linearRegCostFunction(X_poly_test, ytest, theta, 0))
print('Expected test error value (approx) 3.8599')


# 3.5 Optional (ungraded) exercise: Plotting learning curves with randomly selected examples

fig3 = plt.figure()
ax = plt.axes()
error_train, error_val = learningCurve_w_averaging(X_poly, y, X_poly_val, yval, lambda0, iterations, sample_size=50)
plt.plot(1+np.arange(X.shape[0]), error_train, 'b-', label='Train')
plt.plot(1+np.arange(X.shape[0]), error_val, 'g-', label='Cross Validation')
fig3.suptitle('Polynomial Regression Learning Curve with averaging (lambda = {:f})'.format(lambda0))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
ax.set_xlim((0,13))
ax.set_ylim((0,100))
plt.legend()
plt.show()

