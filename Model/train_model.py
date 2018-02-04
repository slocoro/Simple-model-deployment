#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:49:16 2018

@author: Steven
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# import some data to play with
iris = datasets.load_iris()

# separate predictors from outcome
X = iris.data
y = iris.target

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=123)

# train model
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# predict on test set
#neigh.predict(X_test)

# pickle model for later use
filename = 'Iris_model_v1.pkl'
pickle.dump(neigh, open(filename, 'wb'))



# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)
#print(result)


