# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:02:17 2020

@author: hierarchical clustering OMAR
"""

# %reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# using the dendrogram to find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward')) # try to minimize the variance in each clusters
plt.title('dendrogram')
plt.xlabel('costumers')
plt.ylabel('euclidean distance')
plt.show()

# fitting hierarchical clustering into the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'standard')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'carefull')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'pink', label = 'sensible')
plt.title('K-Means Clustering')
plt.xlabel('annual income K$')
plt.ylabel('spending score (1-100)')
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.show()