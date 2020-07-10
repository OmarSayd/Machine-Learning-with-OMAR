# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:14:30 2020

@author: OMAR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values    
y = dataset.iloc[:, 2].values

"""
# since the data set is not large enough to make training and testing set
#spliting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#feature scaling, because of euclidean distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
# Fitting Linear reg to data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting polynomial reg to data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualization for linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='yellow')
plt.title('TRUTH or BLUFF (Linear Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()

# Visualization for polynomial regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='yellow')
plt.title('TRUTH or BLUFF (polynomial Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()
"""
# predicting new result with linear model
lin_reg.predict(6.5)
# predicting new result with Polynomial model
lin_reg_2.predict(poly_reg.fit_transform(6.5))

NOT WORKING
"""
"""
Conclusion: The new employee is telling truth that he was getting $160K for
            6.5 level
"""




