# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:36:28 2020

@author: Apriori Algorithm by OMAR
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)] )

# Training apriori model
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
           # considering min 3 product sold out in a day for a week, so min_support=0.0028


# visualizing the results
results = list(rules)

# NOTE: the answer is not showing the goods properly