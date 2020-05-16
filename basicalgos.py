#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:51:29 2020

@author: shantanu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression, LassoLars, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
import warnings


def basicalgos(X_train, X_test, y_train, y_test, params):
    ridge_paramgrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    lasso_paramgrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    lassolars_paramgrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    elasticnet_paramgrid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "l1_ratio": np.arange(0.0, 1.0, 0.2)}

    knn_paramgrid = {"n_neighbors": list(range(3, 100, 2)),
                     "weights": ['uniform', 'distance'],
                     "p": [1, 2]}

    decision_tree_paramgrid = {'max_depth': [None, 2, 5, 9],
                               'min_samples_split': [2, 5, 9, 13],
                               'min_samples_leaf': [2, 5, 9, 13],
                               'random_state': [123],
                               'max_leaf_nodes': [2, 5, 9, 13]}

    linear_regression = LinearRegression()

    ridge = GridSearchCV(estimator=Ridge(), param_grid=ridge_paramgrid, n_jobs=10, cv=5)

    lasso = GridSearchCV(estimator=Lasso(), param_grid=lasso_paramgrid, n_jobs=10, cv=5)

    lasso_lars = GridSearchCV(estimator=LassoLars(), param_grid=lassolars_paramgrid, n_jobs=10, cv=5)

    elasticnet = GridSearchCV(estimator=ElasticNet(), param_grid=elasticnet_paramgrid, n_jobs=10, cv=5)

    knn = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=knn_paramgrid, n_jobs=10, cv=5)

    decisiontree = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=decision_tree_paramgrid, n_jobs=10, cv=5)

    models = [ridge, lasso, lasso_lars, elasticnet, knn, decisiontree]
    models_names = ["ridge", "lasso", "lasso_lars", "elasticnet", "knn", "decisiontree"]

    for i in range(6):
        print(i + 1, "==============================================================================================")
        print(models_names[i], "training started, Please wait.............")
        print(models_names[i], "details:")
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        print('RMSE in', models_names[i], "-", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        print("---------------------")
        print("Best parameters of", models_names[i], ":")
        print(models[i].best_params_)
        print("---------------------")
        print("Score of", models_names[i], "is", models[i].score(X_test, y_test))
        print("---------------------")
        print("Training Finished")
        print("================================================================================================")
        params["algorithms"][models_names[i]] = (models[i].score(X_test, y_test))*100

    params["algokeys"] = list(params["algorithms"].keys())
    params["algovalues"] = list(params["algorithms"].values())
    params["best_acc"] = round(max(params["algovalues"]),2)
