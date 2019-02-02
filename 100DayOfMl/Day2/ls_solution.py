#compute the least square  solution for 2d data
# ML Machine Learning: A Bayesian and Optimization Perspective chapter 3 example 3.2

import numpy as np
# y= thta0 +theta1*x0 + theta2*x2 +n
# y= 0.25 -0.25*x1 + 0.2*x2 +n
N=5000
mu, sigma = 1, 1 # mean and standard deviation
n = np.random.normal(mu, sigma, N)
x1 = np.random.uniform(0,10,N)
x2 = np.random.uniform(0,10,N)
y = 0.25- 0.25*x1 +0.2*x2 +n
X = np.array([np.ones(N),x1, x2]).transpose()
Y = y.transpose()
theta =np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
theta2 =np.linalg.lstsq(X,Y)
print (theta) #  psuedo inverse
print(theta2)


pass

