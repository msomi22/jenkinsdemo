# Import the Linear Regression module from sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
 
# Need a training dataset, the model will learn how to add numbers from these data
# We only need 3 training examples:
#   2+3=5   #   1+5=6   #   6+5=11
X = [[2,3],[1,5],[5,6]]
Y = [5,6,11]
 
# Fit the linear regression model with the training data
model = LinearRegression()
model.fit(X,Y)
 
# Done! Now we can use predict to sum two numbers
 
# Sum 6 and 6
print "6 + 6 = %d" %model.predict([[6,6]])  #  array.reshape(6, 6)
# Sum 25 and 50
print "25 + 50 = %d" %model.predict([[25,50]])

print "1 + 50 = %d" %model.predict([[1,50]])
print "600 + 50 = %d" %model.predict([[600,50]])
print "333 + 111 = %d" %model.predict([[333,111]])
print "502 + 8 = %d" %model.predict([[502,8]])

print "600 + 400 = %d" %model.predict([[600,400]])
print "1500 + 27 = %d" %model.predict([[1500,27]])
 
# Use this to get rid of Warnings at execution
#print "6 + 6 = %d" %model.predict(np.array([6,6]).reshape(1,-1))
#print "25 + 50 = %d" %model.predict(np.array([25,50]).reshape(1,-1))