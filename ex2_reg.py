'''
Regularised logistic regression. Generating polynomial features. Plotting the decision boundary.

Bence Mélykúti
22-23/02/2018
'''

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# built-in version (recommended)
# http://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features
def mapFeature(X1, X2, degree): # X1 and X2 must be 1-dim vectors, >=2-dim matrices not allowed
# The first column of output is all 1.
    X1=np.asarray(X1)
    X1=X1.reshape((X1.ravel().shape[0],1))
    X2=np.asarray(X2)
    X2=X2.reshape((X2.ravel().shape[0],1))
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(np.concatenate((X1,X2),axis=1))

# own implementations, not really recommended or necessary:
def mapFeature_manual(X1, X2, degree): # X1 and X2 must be 1-dim vectors, >=2-dim matrices not allowed
    X1=np.asarray(X1)
    X1=X1.reshape((X1.ravel().shape[0],1))
    X2=np.asarray(X2)
    X2=X2.reshape((X2.ravel().shape[0],1))
# no. of monomials = (degree+1)*(degree+2)/2 - 1 (-1 is there for constant 1, which is added back later)
    out=np.ones((X1.ravel().shape[0], int((degree+1)*(degree+2)/2-1)))
    for i in range(1, degree+1):
        for j in range(i+1):
            out[:, int(i*(i+1)/2)+j-1] = np.ravel(np.power(X1, i-j)*np.power(X2, j))
    out = np.concatenate((np.ones((X1.ravel().shape[0],1)),out), axis=1)
    return out

def unittest_mapFeature_manual(degree): # for mapFeature_manual(X1, X2)
    print('Unit test of polynomial terms indexing...')
    v=[]
    for i in range(1, degree+1):
        for j in range(i+1):
            vnew=int(i*(i+1)/2)+j-1
            v=v+[vnew]
            print('- - - Entering X1^{0} X2^{1} into column {2}.'.format(i-j, j, vnew))
    if all(pair[0]==pair[1] for pair in zip(v,range(int((degree+1)*(degree+2)/2-1)))):
        print('...column index continuity passed.')
    else:
        print('Error: column index continuity failed.')

# own, supposedly fool-proof implementation, not really recommended or necessary:
def mapFeature_fp(X1, X2, degree): # X1 and X2 must be 1-dim vectors, >=2-dim matrices not allowed
    X1=np.asarray(X1)
    X1=X1.reshape((X1.ravel().shape[0],1))
    X2=np.asarray(X2)
    X2=X2.reshape((X2.ravel().shape[0],1))
    out=np.ones((X1.ravel().shape[0],0))
    for i in range(degree):
        for j in range(i+2):
            out=np.concatenate((out, np.power(X1, i+1-j)*np.power(X2, j)), axis=1)
    out = np.concatenate((np.ones((X1.ravel().shape[0],1)),out), axis=1)
    return out

def unittest_mapFeature_comparison():
    X1=[0,2,4]
    X2=[1,3,5]

    print('These three outputs must be identical:')
    out=mapFeature(X1,X2,2)
    print(out)
    out=mapFeature_manual(X1,X2,2)
    print(out)
    out=mapFeature_fp(X1,X2,2)
    print(out)

def plotDecisionBoundary(theta, X, y, degree): # The first column of input X is all 1. See also ex6.py for an updated version for SVM.

# The decision boundary in the linear case:
# lreg.intercept_[0] + e[0]*lreg.coef_[0][0] + e[1]*lreg.coef_[0][1] == 0  <==>
# e[1] = -(lreg.intercept_[0] + e[0]*lreg.coef_[0][0])/lreg.coef_[0][1]
# There should be a separation of cases to deal with lreg.coef_[0][1]==0, but we assume this never happens.

    plt.scatter(X[y.astype(int).ravel()==1,1], X[y.astype(int).ravel()==1,2], marker="+", c='k', label='y=1')
    plt.scatter(X[y.astype(int).ravel()==0,1], X[y.astype(int).ravel()==0,2], marker="o", c='y', edgecolors='k', label='y=0')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    if X.shape[1]<=3: # linear decision boundary, in this case degree=1
        a = np.array([np.min(X[:,1]), np.max(X[:,1])])
        b = np.divide(-np.add(np.full_like(a, lreg.intercept_[0]), lreg.coef_[0][0]*a), lreg.coef_[0][1])
        plt.plot(a, b, color='blue', linewidth=3, label='Decision boundary')
    else: # non-linear decision boundary, in this case degree>1
        span1=np.max(X[:,1])-np.min(X[:,1])
        span2=np.max(X[:,2])-np.min(X[:,2])
        '''
        # np.arange is also an option:
        linspace1=np.arange(np.min(X[:,1])-span1*1.1, np.max(X[:,1])+span1*1.1, np.ceil(20/span1))
        linspace2=np.arange(np.min(X[:,2])-span2*1.1, np.max(X[:,2])+span2*1.1, np.ceil(20/span2))
        '''
        linspace1=np.linspace(np.min(X[:,1])-span1*0.1, np.max(X[:,1])+span1*0.1, num=100, endpoint=True)
        linspace2=np.linspace(np.min(X[:,2])-span2*0.1, np.max(X[:,2])+span2*0.1, num=100, endpoint=True)
        x1, x2 = np.meshgrid(linspace1, linspace2)
        yy = np.zeros_like(x1)
        for i in range(x1.shape[1]):
            yy[:,i] = np.dot(mapFeature(x1[:,i], x2[:,i], degree), theta)
        plt.contour(x1, x2, yy, levels=[0])

    plt.legend()
    plt.show()


# Unit tests for my own polynomial feature mappings
#unittest_mapFeature_manual(4)
#unittest_mapFeature_comparison()


csvfile = open('../machine-learning-ex2/ex2/ex2data2.txt', 'r', newline='')
csvdata = csv.reader(csvfile, delimiter=',')
dataiter, dataiter2 = itertools.tee(csvdata) # https://docs.python.org/3/library/itertools.html
length=len(next(dataiter))
data=np.empty([0,length])
for row in dataiter2:
    data=np.vstack((data, [float(entry) for entry in row]))

X = data[:,:2]
y = data[:,2]

degree=6
X=mapFeature(X[:,0], X[:,1], degree)

# Visualisation

plt.scatter(data[y.astype(int)==1,0],data[y.astype(int)==1,1], marker="+", c='k', label='y=1')
plt.scatter(data[y.astype(int)==0,0],data[y.astype(int)==0,1], marker="o", c='y', edgecolors='k', label='y=0')
# fig.suptitle('Population-profit plot', fontsize=20)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()


# Regularised logistic regression

# regularisation parameter
lambda0 = 1 # set to tiny value (e.g. 1e-10) to achieve no regularisation

lreg = linear_model.LogisticRegression(C=1/lambda0, fit_intercept=True, intercept_scaling=100)
#    [intercept_scaling] Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.
#    Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

lreg.fit(X[:,1:],y) # the first column with all 1s is taken care of by fit_intercept=True
print('\nTheta found by logistic regression:')
print(lreg.intercept_, lreg.coef_)

data_prediction = lreg.predict(X[:,1:]) # the first column with all 1s was taken care of by fit_intercept=True

print('{0} data points out of {1} correctly classified.'.format(np.sum(y==data_prediction),X.shape[0]))
print('Train Accuracy:')
print(np.mean(y==data_prediction) * 100)
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

plotDecisionBoundary(np.concatenate((np.asarray(lreg.intercept_).reshape(1), lreg.coef_[0])), X, y, degree)

