#!/usr/bin/env python
import sys
import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt 
import pandas
from pandas.plotting import scatter_matrix
import sklearn 
from sklearn  import model_selection 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 


def showVersions():
	#print'python: {}',format(sys.version)
	print'scipy: {}',format(scipy.__version__)
	print'numpy: {}',format(numpy.__version__)
	print'matplotlib: {}',format(matplotlib.__version__)
	print'pandas: {}',format(pandas.__version__)
	print'sklearn: {}',format(sklearn.__version__) 

def readData():
	url = "../data/iris2.csv" 
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url,names=names) 
	#sammarizeData(dataset)
	#plotData(dataset) 
	model_eval(dataset)

'''
Here we do a statistical summary of the data. like shape, head, description etc 
'''
def sammarizeData(dataset):
	#print(dataset.shape) 
	#print(dataset.head(20)) 
	#print(dataset.describe()) 
	print(dataset.groupby('class').size())  

'''
 Data visualization  , box and whisker plots, histograms, scatter plot matrix, 
'''
def plotData(dataset):
	#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	#plt.show() 
	#----------------------------------------------------------------------
	#dataset.hist()
	#plt.show() 
	#-----------------------------------------------------------------------
	#scatter_matrix(dataset)
	#plt.show() 
	print('visualization ... ') 

'''
Models evaluation 
'''
def model_eval(dataset):
	#split validation dataset 
	array = dataset.values 
	X = array[:,0:4] # sepal-length, sepal-width, petal-length, petal-width
	Y = array[:,4]   # class
	validation_size = 0.20 
	seed = 7
	scoring = 'accuracy'
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	#compare_model(X, Y, X_train, Y_train, validation_size, seed, scoring)
	pred_model_knn(X_train, X_validation, Y_train, Y_validation) 
	#pred_model_svm(X_train, X_validation, Y_train, Y_validation) 
 

def compare_model(X, Y, X_train, Y_train, validation_size, seed, scoring):	 
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC())) 
	results = []
	names = [] 
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold,scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg) 

	#compare model on a graph   
	#compare_model_grph(results, names)	



def compare_model_grph(results, names):
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show() 


#K-Nearest Neighbors 
def pred_model_knn(X_train, X_validation, Y_train, Y_validation):
	print('************ -KNN- ********************************')
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	'''
	my_x_validation = [[5.1, 3.5, 1.4, 0.2],[4.9, 3, 1.4, 0.2],[4.7, 3.2, 1.3, 0.2],[7, 3.2, 4.7, 1.4],[7,3,4,1]]
	my_x_train = numpy.array(my_x_validation) + 0. # This is a numpy floating array
	my_y_validation = ['Iris-setosa','Iris-setosa','Iris-setosa','Iris-versicolor','Iris-versicolor'] 
	#print my_y_validation
	predictions = knn.predict(my_x_train)  
	'''
	predictions = knn.predict(X_validation) 
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
	'''
	print(accuracy_score(my_y_validation, predictions))
	print(confusion_matrix(my_y_validation, predictions))
	print(classification_report(my_y_validation, predictions))
	'''

#Support Vector Machines
def pred_model_svm(X_train, X_validation, Y_train, Y_validation):
	print('************ -SVM- ********************************') 
	svm = SVC() 
	svm.fit(X_train,Y_train)
	predictions = svm.predict(X_validation) 
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))






#showVersions() 
#eadData() 

# Statistical Summary
import pandas
url = "../data/mult.csv" 
names = ['num1', 'num2', 'product']
data = pandas.read_csv(url, names=names)
'''
description = data.describe()
print(description)
'''
array = data.values 
X = array[:,0:2] # sepal-length, sepal-width, petal-length, petal-width
Y = array[:,2]   # class

validation_size = 0.25 
seed = 52 #numpy.random.seed()
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)



my_x_validation = [[3,3]] 
my_x_train = numpy.array(my_x_validation) 
my_y_validation = [9]  


predictions = knn.predict(my_x_validation) 


print('-----------------------------')
print X_train
print('-----------------------------')
print Y_train
print('-----------------------------')


print my_x_validation , ' --> ',  predictions

