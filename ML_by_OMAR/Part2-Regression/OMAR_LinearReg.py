# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:55:52 2020

@author: OMAR
"""

# Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values    #[:,0:3]
y = dataset.iloc[:,1].values


#spliting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,random_state=0)
"""
#feature scaling, because of euclidean distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

not required for simple linear regression
"""
# fitting simple regression model to the training set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,y_train)

#predicting the test set result
y_pred = Regressor.predict(X_test)

#visvualization of training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, Regressor.predict(X_train), color='yellow')
plt.title('salary vs experiance: (training set)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()

#visvualization of test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, Regressor.predict(X_train), color='blue')
plt.title('salary vs experiance: (test set)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()













