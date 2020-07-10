# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:40:34 2020

@author: ThompsonSampling OMAR
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_reward = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_reward:
            max_reward = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected, color = 'green')
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()