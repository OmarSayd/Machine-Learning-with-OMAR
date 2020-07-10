# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:04:21 2020

@author: OMAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values    
y = dataset.iloc[:, 2:3].values  # 2:3 for converting a vector into matrix

"""
# since the data set is not large enough to make training and testing set
#spliting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)"""

#feature scaling, because of euclidean distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualization for regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='yellow')
plt.title('TRUTH or BLUFF (SVR Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()

# Visualization SVR regression result for higher resolution and smother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='yellow')
plt.title('TRUTH or BLUFF (SVR Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()