# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:03:35 2018

@author: gk
"""

#For the KAGGLE data, looks like most series (~2/3) are dense [no sparsity]
#important because in Arturius's script there is threshold on #0's allowed, default he seems to use is not allow any 0's
#so then he is using ~2/3 of the time series ???

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




filepath = r"/......./kaggle-web-traffic-master/data/train_1.csv"

df = pd.read_csv(filepath)

rows = df.values

x = [(i>0.).sum() -1 for i in rows]
ndays = max(x)
x = [float(i) / float (ndays) for i in x]

x.sort()

#Sorted plot of percent dense [so about 2/3 of the 145K Kaggle are dense]
plt.figure()
plt.plot(x)
plt.show()