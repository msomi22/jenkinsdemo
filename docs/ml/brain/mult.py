# Import the Linear Regression module from sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
 
# Need a training dataset, the model will learn how to add numbers from these data
# We only need 3 training examples:
#   2+3=5   #   1+5=6   #   6+5=11
X = [[4,4],[6,5],[7,7],[10,7],[50,6],[8,6],[40,8],[37,9],[6,6],[30,2],[600,5]]
Y = [16,30,49,70,300,48,320,333,36,60,3000] 

# Fit the linear regression model with the training data
model = LinearRegression()
model.fit(X,Y)
 
# Done! Now we can use predict to sum two numbers
 
# Sum 6 and 6
print "6 * 6 = %d" %model.predict([[6,6]])  #  array.reshape(6, 6)
# Sum 25 and 50
print "30 * 2 = %d" %model.predict([[30,2]]) 


print "10 * 10 = %d" %model.predict([[10,10]])   
print "150 * 2 = %d" %model.predict([[150,2]])  
print "500 * 5 = %d" %model.predict([[500,5]])  
 
# Use this to get rid of Warnings at execution
#print "6 + 6 = %d" %model.predict(np.array([6,6]).reshape(1,-1))
#print "25 + 50 = %d" %model.predict(np.array([25,50]).reshape(1,-1))