# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:32:33 2020

@author: KMeans OMAR
"""
#%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

# using the elbow method to find the optimul no. of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300,random_state=101)
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)     # .inertia_ is used as wcss
plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('No. Of cluster')
plt.ylabel('WCSS')
plt.show()
# so the no. of cluster is 5

# applying kmeans into the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=101)
y_kmeans = kmeans.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'standard')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'carefull')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'orange', label = 'careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'pink', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'centroid')
plt.title('K-Means Clustering')
plt.xlabel('annual income K$')
plt.ylabel('spending score (1-100)')
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.show()