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


# Fitting regression model to dataset

# create your regressor here

# predicting a new result
y_pred = regressor.predict(6.5)

# Visualization for regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='yellow')
plt.title('TRUTH or BLUFF (Linear Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()

# Visualization regression result for higher resolution and smother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='yellow')
plt.title('TRUTH or BLUFF (Regression Model)')
plt.xlabel('Level')
plt.ylabel('salary')
plt.show()

