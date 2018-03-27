'''
Linear regression with multiple variables. Feature normalisation.

Bence Mélykúti
22/02/2018
'''

import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing

csvfile = open('../machine-learning-ex1/ex1/ex1data2.txt', 'r', newline='')
csvdata = csv.reader(csvfile, delimiter=',')
dataiter, dataiter2 = itertools.tee(csvdata) # https://docs.python.org/3/library/itertools.html
length=len(next(dataiter))
data=np.empty([0,length])
for row in dataiter2:
    data=np.vstack((data, [float(entry) for entry in row]))

X = data[:,:2]
X_orig = X
y = data[:,2]
m = len(y)


# https://matplotlib.org/2.0.2/api/pyplot_api.html
# fig = plt.figure()
plt.scatter(X[:,0], y, marker="x", c='r')
# fig.suptitle('Size-price plot', fontsize=20)
plt.ylabel('Price')
plt.show()


# Own implementation of feature normalisation; just use sklearn.preprocessing.StandardScaler instead
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    mu_matrix = np.concatenate([mu for i in range(X.shape[0])], axis=0)
    sigma_matrix = np.concatenate([sigma for i in range(X.shape[0])], axis=0)
    X = (X - mu)/sigma
    return X, mu, sigma

Xsc_own, mu_own, sigma_own = featureNormalize(X)


# feature normalisation with sklearn.preprocessing.StandardScaler:
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = preprocessing.StandardScaler().fit(X)
Xsc = scaler.transform(X) # scaled X
mu = scaler.mean_
sigma2 = scaler.var_ # variance, i.e. sigma^2
sigma = scaler.scale_ # standard deviation, i.e. sigma


# Linear regression, classic fitting

# Linear regression example
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

reg = linear_model.LinearRegression()
reg.fit(Xsc,y) # for X already scaled from featureNormalize or StandardScaler
# reg.fit(scaler.transform(X),y)
print('\nTheta found by linear regression:')
print(reg.intercept_, reg.coef_)

data_prediction = reg.predict(Xsc) # for X already scaled from featureNormalize or StandardScaler
#data_prediction = reg.predict(scaler.transform(X))

# These four are equivalent:
#predict1 = reg.predict((((1650,3)-mu_own)/sigma_own).reshape(1,-1)) # using featureNormalize
#predict1 = reg.predict((((1650,3)-mu)/sigma).reshape(1,-1)) # using output of sklearn.preprocessing.StandardScaler
#predict1 = reg.predict((((1650,3)-mu)/sigma2**0.5).reshape(1,-1)) # using output of sklearn.preprocessing.StandardScaler
predict1 = reg.predict(scaler.transform(np.asarray((1650,3)).astype(float).reshape(1,-1))) # recommended way of using sklearn.preprocessing.StandardScaler
print('Predicted price of a 1650 sq-ft, 3 br house: ',predict1[0])


# Linear regression, normal equation

Xe = np.concatenate((np.ones((X.shape[0],1)), Xsc), axis=1)
theta=np.dot(np.linalg.inv(np.dot(np.transpose(Xe),Xe)), np.dot(np.transpose(Xe), y))
print('\nTheta found by normal equation:')
print(theta)

'''
# In comparison to ex1.py, there is no point in plotting it in 2 dim, as there are two features and one value, together 3 dim:
# https://matplotlib.org/2.0.2/api/pyplot_api.html
# fig = plt.figure()
plt.scatter(X[:,1],y, marker="x", c='r')
# fig.suptitle('Population-profit plot', fontsize=20)
plt.plot(X[:,1], data_prediction, color='blue', linewidth=3)
plt.ylabel('Price')
plt.show()
'''
