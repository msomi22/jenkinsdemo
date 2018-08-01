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



url = "data/WIKI-PRICES.csv"  
#names = ['open', 'high', 'close', 'volume', 'adj_open']
dataset = pandas.read_csv(url,names=None) 

#print(dataset.shape) 
#print(dataset.describe()) 
#print(dataset.head()) 

array = dataset.values 
df = array[:,2:6] 
print(df)  

