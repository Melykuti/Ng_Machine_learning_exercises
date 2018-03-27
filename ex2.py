'''
Sigmoid function. Logistic regression. Plotting a linear decision boundary.

Bence Mélykúti
22/02/2018
'''

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model

# https://stackoverflow.com/questions/12514890/python-numpy-test-for-ndarray-using-ndim#12514951
def sigmoid(z): # input can be anything (scalar, tuple, list, np.array), this outputs an np.array
    z = np.asarray(z)
    # These two are equivalent:
    #g = np.divide(np.ones(z.shape), np.ones(z.shape) + np.exp(np.negative(z)))
    g = np.reciprocal(np.ones(z.shape) + np.exp(np.negative(z)))
    return g


csvfile = open('../machine-learning-ex2/ex2/ex2data1.txt', 'r', newline='')
csvdata = csv.reader(csvfile, delimiter=',')
dataiter, dataiter2 = itertools.tee(csvdata) # https://docs.python.org/3/library/itertools.html
length=len(next(dataiter))
data=np.empty([0,length])
for row in dataiter2:
    data=np.vstack((data, [float(entry) for entry in row]))

X = data[:,:2]
y = data[:,2]

# Visualisation

plt.scatter(data[y.astype(int)==1,0],data[y.astype(int)==1,1], marker="+", c='k', label='Admitted')
plt.scatter(data[y.astype(int)==0,0],data[y.astype(int)==0,1], marker="o", c='y', edgecolors='k', label='Not admitted')
# fig.suptitle('Exam scores', fontsize=20)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


# Logistic regression

lreg = linear_model.LogisticRegression(C=1e+10, intercept_scaling=1) # C must be large for minimal regularisation, e.g. 1e+10
lreg.fit(X,y)
print('\nTheta found by logistic regression:')
print(lreg.intercept_, lreg.coef_)
print('Expected theta (approx):')
print(' (-25.161, 0.206, 0.201)')

data_prediction = lreg.predict(X)

# Number of data points correctly classified:
# print(np.sum(y==data_prediction))

# Evaluating the hypothesis h_theta with the fitted theta, substituted into the sigmoid function:
'''
# These two are equivalent:
# By writing it out in detail
neg_lin_comb = np.negative(np.add(np.full((X.shape[0]), lreg.intercept_[0]), np.dot(X, lreg.coef_[0])))
g2 = np.reciprocal(np.ones(X.shape[0]) + np.exp(neg_lin_comb))
'''
# By calling the above defined function sigmoid:
lin_comb = np.add(np.full((X.shape[0]), lreg.intercept_[0]), np.dot(X, lreg.coef_[0]))
g = sigmoid(lin_comb)


# Printing incorrectly classified points (classification, value of sigmoid function, ground truth):
print('\nIncorrectly classified points (classification, value of sigmoid function, ground truth):')
for i in range(X.shape[0]):
    if y[i]!=data_prediction[i]:
        print(int(data_prediction[i]),g[i],int(y[i]))

# Here any combination of e and prob should work:
e=[45, 85]
#e=np.array([45, 85])
#prob=1/(1+np.exp(-(lreg.intercept_[0] + e[0]*lreg.coef_[0][0] + e[1]*lreg.coef_[0][1])))
prob=sigmoid(np.array(lreg.intercept_[0] + e[0]*lreg.coef_[0][0] + e[1]*lreg.coef_[0][1]))

print('\nFor a student with scores 45 and 85, we predict\n\
an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002')


# The decision boundary

# lreg.intercept_[0] + e[0]*lreg.coef_[0][0] + e[1]*lreg.coef_[0][1] == 0  <==>
# e[1] = -(lreg.intercept_[0] + e[0]*lreg.coef_[0][0])/lreg.coef_[0][1]
a = np.array([np.min(X[:,1]), np.max(X[:,1])])
b = np.divide(-np.add(np.full_like(a, lreg.intercept_[0]), lreg.coef_[0][0]*a),lreg.coef_[0][1])


# https://matplotlib.org/2.0.2/api/pyplot_api.html
#fig = plt.figure()
plt.scatter(data[y.astype(int)==1,0],data[y.astype(int)==1,1], marker="+", c='k', label='Admitted')
plt.scatter(data[y.astype(int)==0,0],data[y.astype(int)==0,1], marker="o", c='y', edgecolors='k', label='Not admitted')
plt.plot(a, b, color='blue', linewidth=3, label='Decision boundary')
#fig.suptitle('Exam scores', fontsize=20)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

