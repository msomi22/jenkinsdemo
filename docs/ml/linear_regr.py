'''
Linear Regression for Machine Learning

*belongs to both statistics and machine learning.
*Linear regression is a linear model, e.g. a model 
that assumes a linear relationship between the input variables (x)
 and the single output variable (y). More specifically, 
 that y can be calculated from a linear combination of the 
 input variables (x).
*both the input values (x) and the output value are numeric.
*y = B0 + B1*x
*weight =B0 +B1 * height

regression - continous line, straigt line
y = mx + c

Linear regression assumes that the relationship between your 
input and output is linear. 

Linear Assumption. 
Linear regression assumes that the relationship between your input and output is linear. 
It does not support anything else. This may be obvious, but it is good to remember when you have a lot of attributes. 
You may need to transform data to make the relationship linear (e.g. log transform for an exponential relationship).

Remove Noise. 
Linear regression assumes that your input and output variables are not noisy. 
Consider using data cleaning operations that let you better expose and clarify the signal in your data. 
This is most important for the output variable and you want to remove outliers in the output variable (y) if possible.

Remove Collinearity. 
Linear regression will over-fit your data when you have highly correlated input variables. 
Consider calculating pairwise correlations for your input data and removing the most correlated.

Gaussian Distributions. 
Linear regression will make more reliable predictions if your input and output variables have a Gaussian distribution. 
You may get some benefit using transforms (e.g. log or BoxCox) on you variables to make their distribution more Gaussian looking.

Rescale Inputs: Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.

import Quandl as q
import quandl as q


Your API key is:
gzfHH1v-u5sMjKbE2xwj


'''

import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
from matplotlib import style
import datetime
#import pickle


style.use('ggplot')


quandl.ApiConfig.api_key = 'gzfHH1v-u5sMjKbE2xwj'
df = quandl.get('WIKI/GOOGL')
#df = quandl.get('data/WIKI-PRICES.csv') 
#print(df.head()) 

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head()) 


forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] 
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

'''
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

'''
 
clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
#'''

##print(confidence)
##pickle_in = open('linearregression.pickle','rb')
##clf = pickle.load(pickle_in)



#print(confidence)


forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()







