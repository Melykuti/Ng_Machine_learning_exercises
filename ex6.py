'''
Support vector machines. Plotting the decision boundary. Gaussian kernel (radial basis function kernel). Selecting the regularisation parameter using a cross-validation set.

Bence Mélykúti
26/02/2018
'''

import numpy as np
import scipy.io # to open Matlab's .mat files
import matplotlib.pyplot as plt
from sklearn import svm

def plotData(X, y):
    plt.scatter(X[y.astype(int).ravel()==1,0], X[y.astype(int).ravel()==1,1], marker="+", c='k', label='y=1')
    plt.scatter(X[y.astype(int).ravel()==0,0], X[y.astype(int).ravel()==0,1], marker="o", c='y', edgecolors='k', label='y=0')
    # fig.suptitle('Plot', fontsize=20)
    #plt.xlabel('Test 1')
    #plt.ylabel('Test 2')
    plt.legend()
    plt.show()

def plotData_with_support_vectors(X, y, clf): 
    plt.scatter(X[y.astype(int).ravel()==1,0], X[y.astype(int).ravel()==1,1], marker="+", c='k', label='y=1')
    plt.scatter(X[y.astype(int).ravel()==0,0], X[y.astype(int).ravel()==0,1], marker="o", c='y', edgecolors='k', label='y=0')
    plt.scatter(X[clf.support_,0], X[clf.support_,1], marker=".", c='r', label='Support vectors')
    # fig.suptitle('Plot', fontsize=20)
    #plt.xlabel('Test 1')
    #plt.ylabel('Test 2')
    plt.legend()
    plt.show()

def plotDecisionBoundary(clf, X, y, linear, with_support_vectors): # The first column of input X is all 1.

# The decision boundary in the linear case:
# lreg.intercept_[0] + e[0]*lreg.coef_[0][0] + e[1]*lreg.coef_[0][1] == 0  <==>
# e[1] = -(lreg.intercept_[0] + e[0]*lreg.coef_[0][0])/lreg.coef_[0][1]
# There should be a separation of cases to deal with lreg.coef_[0][1]==0, but we assume this never happens.

    plt.scatter(X[y.astype(int).ravel()==1,1], X[y.astype(int).ravel()==1,2], marker="+", c='k', label='y=1')
    plt.scatter(X[y.astype(int).ravel()==0,1], X[y.astype(int).ravel()==0,2], marker="o", c='y', edgecolors='k', label='y=0')
    if with_support_vectors==1:
        plt.scatter(X[clf.support_,1], X[clf.support_,2], marker=".", c='r', label='Support vectors')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    if linear==1: # linear decision boundary
        a = np.array([np.min(X[:,1]), np.max(X[:,1])])
        b = np.divide(-np.add(np.full_like(a, clf.intercept_[0]), clf.coef_[0][0]*a),clf.coef_[0][1])
        plt.plot(a, b, color='blue', linewidth=2, label='Decision boundary')
    else: # non-linear decision boundary, contour plotted
        span1=np.max(X[:,1])-np.min(X[:,1])
        span2=np.max(X[:,2])-np.min(X[:,2])
        linspace1=np.linspace(np.min(X[:,1])-span1*0.1, np.max(X[:,1])+span1*0.1, num=100, endpoint=True)
        linspace2=np.linspace(np.min(X[:,2])-span2*0.1, np.max(X[:,2])+span2*0.1, num=100, endpoint=True)
        x1, x2 = np.meshgrid(linspace1, linspace2)
        yy = np.zeros_like(x1)
        for i in range(x1.shape[1]):
            yy[:,i] = clf.predict(np.stack((x1[:,i], x2[:,i]), axis=1))
        plt.contour(x1, x2, yy, levels=[0])

    plt.legend()
    plt.show()

def optimalParams(X, y, Xval, yval):
    C_vec=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec=C_vec
    errors=np.zeros((len(C_vec), len(sigma_vec)))
    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            clf = svm.SVC(C_vec[i], kernel='rbf', gamma=1/(2*sigma_vec[j]**2))
            clf.fit(X, y.ravel())
            errors[i,j]=np.mean(clf.predict(Xval)!=yval.ravel())
    #print(errors)
    argminind = np.unravel_index(np.argmin(errors, axis=None), (len(C_vec),len(sigma_vec)))
    return C_vec[argminind[0]], sigma_vec[argminind[1]]


# =============== Part 1: Loading and Visualizing Data ================

data = scipy.io.loadmat('../machine-learning-ex6/ex6/ex6data1.mat')

X = data['X']
y = data['y']

# Visualisation

plotData(X, y)

# ==================== Part 2: Training Linear SVM ====================

C = 1e+2
clf = svm.SVC(C, kernel='linear')
clf.fit(X, y.ravel())

print('clf.support_',clf.support_)
#print('clf.support_vectors_',clf.support_vectors_)
print('clf.n_support_',clf.n_support_)
print('clf.intercept_', clf.intercept_)
print('clf.coef_',clf.coef_)

# Visualisation

# Data with support vectors in red
#plotData_with_support_vectors(X, y, clf)

# Data with support vectors in red and the decision boundary
Xe=np.concatenate((np.ones((X.shape[0],1)),X), axis=1) # prepending a column of 1s
plotDecisionBoundary(clf, Xe, y, 1, 1)

# =============== Part 3: Implementing Gaussian Kernel ===============
# =============== Part 4: Visualizing Dataset 2 ================

data = scipy.io.loadmat('../machine-learning-ex6/ex6/ex6data2.mat')

X = data['X']
y = data['y']

# Visualisation

plotData(X, y)

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# RBF = radial basis function kernel; essentially the Gaussian kernel
# http://scikit-learn.org/stable/modules/svm.html#svm-kernels

C = 1e+0
sigma = 0.1
clf = svm.SVC(C, kernel='rbf', gamma=1/(2*sigma**2))
clf.fit(X, y.ravel())

Xe=np.concatenate((np.ones((X.shape[0],1)),X), axis=1) # prepending a column of 1s
plotDecisionBoundary(clf, Xe, y, 0, 0)

# =============== Part 6: Visualizing Dataset 3 ================

data = scipy.io.loadmat('../machine-learning-ex6/ex6/ex6data3.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']

# Visualisation

plotData(X, y)
#plotData(Xval, yval)

# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

C, sigma = optimalParams(X, y, Xval, yval)
print('Optimal choices: C = {}, sigma = {}'.format(C, sigma))
# If you want overfit, try:
#C, sigma = 3, 0.03
clf = svm.SVC(C, kernel='rbf', gamma=1/(2*sigma**2))
clf.fit(X, y.ravel())


Xe=np.concatenate((np.ones((X.shape[0],1)),X), axis=1) # prepending a column of 1s
plotDecisionBoundary(clf, Xe, y, 0, 0)

Xvale=np.concatenate((np.ones((Xval.shape[0],1)),Xval), axis=1) # prepending a column of 1s
plotDecisionBoundary(clf, Xvale, yval, 0, 0)
