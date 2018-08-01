from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


'''
A simple numeric prediction based on some funny series 
'''
def numericPrediction():
	X = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]]
	Y = [10,20,30,40,50,60,70,80,90,100,110]
	model = LinearRegression()
	model.fit(X,Y)
	print "12 + 12 = %d" %model.predict([[12,12]])
	print "13 + 13 = %d" %model.predict([[13,13]])
	print "20 + 20 = %d" %model.predict([[20,20]])
	print "21 + 21 = %d" %model.predict([[21,21]]) 


#numericPrediction()


'''
Predict area of a circle 
Area = PI * (r * r)
PI = np.pi 

we will first write a simple method to return area of a circle given a particular radius 

'''
def getCircleArea(radius):
	return round(np.pi * radius * radius, 4)

def circleAreaPrediction(predRadius): 
	X = []  
	Y = []
	for num in range(1,14): 
		X.append([num])
		Y.append(getCircleArea(num))#
	
    #Train our model 
	model = LinearRegression()
	model.fit(X,Y) 
	print X
	print Y
	print '******************************************************************************************************'
	print "Circle r = ", predRadius ," A = " , model.predict([[predRadius]]) , " CA = ", getCircleArea(predRadius) 
	print '******************************************************************************************************'
	plt.scatter(X,Y)
	plt.xlabel("Circle Radius")
	plt.ylabel("Cricle Area")
	plt.title("Linear Regression, Circle Area Prediction") 
	plt.savefig('circleAreaLineGraph')   
	#plt.show() 

circleAreaPrediction(7)  
