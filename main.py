#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:37:26 2020

@author: shantanu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from basic import basicalgos
from advance import randomforest, svm, gradientboost
import threading as th
from clean import cleaning

scaler = MinMaxScaler()

data = pd.read_csv('USA_Housing.csv')

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']
X,y = cleaning(X,y)
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

basicalgos(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
randomforest(X_train, X_test, y_train, y_test)
gradientboost(X_train, X_test, y_train, y_test)
#child1 = th.Thread(target=randomforest)
#child2 = th.Thread(target=svm)
#child3 = th.Thread(target=gradientboost)

# child 1 start training of randomforest
# child 2 start training of svm
# child 3 start training of gradientboost
#child1.start()
#child2.start()
#child3.start()
