'''
Linear regression 
'''
import numpy as np 

ALPHA = 0.002
ITERATIONS = 50

# Hypothesis function 
def predict(X, Theta):
	return X.dot(Theta)

# Cost function 
def cost(X, Y, Theta):
	return np.sum((predict(X,Theta) - Y)**2)/(2*len(X))

# Optimization function 
def descend(X, Y, Theta):
	return Theta - ALPHA * (np.mean((predict(X, Theta) - Y) * X.T, axis=1))

def init_weights(X):
	return np.random.uniform(-1,1,X.shape[1])


Theta = init_weights(X) 	

costs = []
for i in range(ITERATIONS):
	Theta = descend(X,Y,Theta)
	costs.append(cost(X,Y,Theta)) 

#print 'Theta : ' , Theta 
