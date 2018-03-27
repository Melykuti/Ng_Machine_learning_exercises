'''
Linear regression with one variable, ridge regression (L2 regularisation), linear regression with stochastic gradient descent. Plotting.

Bence Mélykúti
22/02/2018
'''

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model

csvfile = open('../machine-learning-ex1/ex1/ex1data1.txt', 'r', newline='')
csvdata = csv.reader(csvfile, delimiter=',')
dataiter, dataiter2 = itertools.tee(csvdata) # https://docs.python.org/3/library/itertools.html
length=len(next(dataiter))
data=np.empty([0,length])
for row in dataiter2:
    data=np.vstack((data, [float(entry) for entry in row]))

# https://matplotlib.org/2.0.2/api/pyplot_api.html
# fig = plt.figure()
plt.scatter(data[:,0], data[:,1], marker="x", c='r')
# fig.suptitle('Population-profit plot', fontsize=20)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

print('With theta = [0 ; 0]\nCost computed =', np.sum(np.square(0+0*data[:,0]-data[:,1]))/(2*data.shape[0]))
print('Expected cost value (approx) 32.07\n')

print('With theta = [-1 ; 2]\nCost computed =', np.sum(np.square(-1+2*data[:,0]-data[:,1]))/(2*data.shape[0]))
print('Expected cost value (approx) 54.24\n')

print('Expected theta values in gradient descent (approx) -3.6303, 1.1664')


# Linear regression, classic fitting

# Linear regression example
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

reg = linear_model.LinearRegression()
reg.fit(data[:,0,np.newaxis],data[:,1,np.newaxis])
print('\nTheta found by linear regression:')
print(reg.intercept_, reg.coef_)

data_prediction = reg.predict(data[:,np.newaxis,0])

predict1 = reg.predict(3.5)
print('For population = 35,000, we predict a profit of ',predict1[0,0]*10000)
predict2 = reg.predict(7)
print('For population = 70,000, we predict a profit of ',predict2[0,0]*10000)



# Linear regression, ridge regression
iterations = 1500
alpha = 0 # regularisation strength
ridge = linear_model.Ridge(alpha=alpha, fit_intercept=True, max_iter=iterations, tol=1e-5)
ridge.fit(data[:,0,np.newaxis],data[:,1,np.newaxis])
print('\nTheta found by ridge regression:')
print(ridge.intercept_, ridge.coef_)

data_prediction = ridge.predict(data[:,np.newaxis,0])


# Linear regression, stochastic gradient descent

# Stochastic gradient descent classification example
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html

iterations = 1500
alpha = 0.001
sgdreg = linear_model.SGDRegressor(loss='squared_loss', penalty='none', learning_rate='constant', eta0=alpha, fit_intercept=True, max_iter=iterations, tol=None, shuffle=True)
sgdreg.fit(data[:,0,np.newaxis],data[:,1,np.newaxis])
print('\nTheta found by stochastic gradient descent:')
print(sgdreg.intercept_, sgdreg.coef_)

data_prediction = sgdreg.predict(data[:,np.newaxis,0])


# https://matplotlib.org/2.0.2/api/pyplot_api.html
# fig = plt.figure()
plt.scatter(data[:,0],data[:,1], marker="x", c='r')
# fig.suptitle('Population-profit plot', fontsize=20)
plt.plot(data[:,0], data_prediction, color='blue', linewidth=3)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

