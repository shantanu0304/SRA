#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:00:06 2020

@author: shantanu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge,LinearRegression,LassoLars, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
import warnings

random_forest_paramgrid_best = dict()
gb_paramgrid_best = dict()
svm_paramgrid_best = dict()

def randomforest(X_train, X_test, y_train, y_test,params):
    print("RandomForest Training Started, Please wait.........")
    rf_choose = {'criterion': ['mse'],
                 'max_depth': list(range(2,100,2)),
                 'max_features':['auto', 'sqrt','log2', None],
                 'min_samples_leaf': list(range(2,100,2)),
                 'min_samples_split': list(range(2,100,2))}
    
    random_forest_paramgrid = {'criterion': hp.choice('criterion', ['mse']),
                               'max_depth': hp.choice('max_depth', list(range(2,100,2))),
                               'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
                               'min_samples_leaf': hp.choice('min_samples_leaf', list(range(2,100,2))),
                               'min_samples_split' : hp.choice('min_samples_split', list(range(2,100,2)))}
    
    def objective_rf(random_forest_paramgrid):
        model = RandomForestRegressor(criterion = random_forest_paramgrid['criterion'],
                                      max_depth = random_forest_paramgrid['max_depth'],
                                      max_features = random_forest_paramgrid['max_features'],
                                      min_samples_leaf = random_forest_paramgrid['min_samples_leaf'],
                                      min_samples_split = random_forest_paramgrid['min_samples_split'],
                                      n_estimators = 200,
                                      random_state=123)
        
        accuracy = cross_val_score(model, X_train, y_train, cv = 4).mean()
        
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials_rf = Trials()
    best_rf = fmin(fn= objective_rf,
                   space= random_forest_paramgrid,
                   algo= tpe.suggest,
                   max_evals = 100,
                   trials= trials_rf)
    
    print("best_rf:")
    print(best_rf)
    
    for i in best_rf.keys():
        random_forest_paramgrid_best[i] = rf_choose[i][best_rf[i]]
    
    print("random_forest_paramgrid_best:")
    print(random_forest_paramgrid_best)
    
    model = RandomForestRegressor(criterion = random_forest_paramgrid_best['criterion'], 
                                  max_depth = random_forest_paramgrid_best['max_depth'],
                                  max_features = random_forest_paramgrid_best['max_features'],
                                  min_samples_leaf = random_forest_paramgrid_best['min_samples_leaf'],
                                  min_samples_split = random_forest_paramgrid_best['min_samples_split'],
                                  n_estimators = 200,
                                  random_state=123)
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    print('RMSE -',np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print("---------------------")
    print("Score - ",metrics.r2_score(y_test,predictions))

    params['algorithms']["randomforest"] = (metrics.r2_score(y_test,predictions))*100

    params["algokeys"] = list(params["algorithms"].keys())
    params["algovalues"] = list(params["algorithms"].values())

    if params["best_acc"] < round(params['algorithms']["randomforest"], 2):
        params["best_acc"] = round(params['algorithms']["randomforest"], 2)

#------------------------------------------------------------------------------------------------------------------------

def gradientboost(X_train, X_test, y_train, y_test,params):
    print("Gradient Boost Training Started, Please wait.........")
    gb_choose = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                 'learning_rate': [0.001, 0.01, 0.1, 1, 10],
                 'max_depth': list(range(2,50,2)),
                 'min_samples_leaf': list(range(2,50,2)),
                 'min_samples_split': list(range(2,50,2)),
                 'max_leaf_nodes': list(range(2,50,2))}
    gb_paramgrid = {'loss': hp.choice('loss',['ls', 'lad', 'huber', 'quantile']),
                    'learning_rate': hp.choice('learning_rate',[0.001, 0.01, 0.1, 1, 10]),
                    'max_depth': hp.choice('max_depth',list(range(2,50,2))),
                    'min_samples_leaf': hp.choice('min_samples_leaf', list(range(2,50,2))),
                    'min_samples_split' : hp.choice('min_samples_split', list(range(2,50,2))),
                    'max_leaf_nodes': hp.choice('max_leaf_nodes',list(range(2,50,2)))}
    
    def objective_gb(gb_paramgrid):
        model = GradientBoostingRegressor(loss = gb_paramgrid['loss'],
                                          max_depth = gb_paramgrid['max_depth'],
                                          learning_rate = gb_paramgrid['learning_rate'],
                                          min_samples_leaf = gb_paramgrid['min_samples_leaf'],
                                          min_samples_split = gb_paramgrid['min_samples_split'],
                                          max_leaf_nodes= gb_paramgrid['max_leaf_nodes'],
                                          random_state=123,
                                          n_estimators=200)
    
        accuracy = cross_val_score(model, X_train, y_train, cv = 4).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK }
    
    trials_gb = Trials()
    best_gb = fmin(fn= objective_gb,
                   space= gb_paramgrid,
                   algo= tpe.suggest,
                   max_evals = 100,
                   trials= trials_gb)
    print("best_gb:")
    print(best_gb)
    
    for i in best_gb.keys():
        gb_paramgrid_best[i] = gb_choose[i][best_gb[i]]
    print("gb_paramgrid_best:")
    print(gb_paramgrid_best)

    model = GradientBoostingRegressor(loss = gb_paramgrid_best['loss'],
                                      max_depth = gb_paramgrid_best['max_depth'],
                                      learning_rate = gb_paramgrid_best['learning_rate'],
                                      min_samples_leaf = gb_paramgrid_best['min_samples_leaf'],
                                      min_samples_split = gb_paramgrid_best['min_samples_split'],
                                      max_leaf_nodes= gb_paramgrid_best['max_leaf_nodes'],
                                      random_state=123,
                                      n_estimators=200)

    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print('RMSE -',np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print("---------------------")
    print("Score - ",metrics.r2_score(y_test,predictions))

#------------------------------------------------------------------------------------------------------------------------

def svm(X_train, X_test, y_train, y_test,params):
    print("SVM Training Started, Please wait.........")
    svm_choose = {'C': [0.0001,0.001,0.01, 0.1, 1.0, 10.0, 100.0,1000.0,10000],
                  'gamma': ['scale','auto'], 
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 

    svm_paramgrid = {'C': hp.choice('C',[0.0001,0.001,0.01, 0.1, 1.0, 10.0, 100.0,1000.0,10000.0]),  
                     'gamma': hp.choice('gamma',['scale','auto']), 
                     'kernel': hp.choice('kernel',['linear', 'poly', 'rbf', 'sigmoid'])}
    
    def objective_svm(svm_paramgrid):
        model = SVR(C = svm_paramgrid['C'],
                    gamma = svm_paramgrid['gamma'],
                    kernel = svm_paramgrid['kernel'])
    
        accuracy = cross_val_score(model, X_train, y_train, cv = 4).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK }

    trials_svm = Trials()
    best_svm = fmin(fn= objective_svm,
                    space= svm_paramgrid,
                    algo= tpe.suggest,
                    max_evals = 100,
                    trials= trials_svm)
    print("best_svm")
    print(best_svm)
    
    for i in best_svm.keys():
        svm_paramgrid_best[i] = svm_choose[i][best_svm[i]]
    
    print("svm_paramgrid_best")
    print(svm_paramgrid_best)

    model = SVR(C = svm_paramgrid_best['C'],
                gamma = svm_paramgrid_best['gamma'],
                kernel = svm_paramgrid_best['kernel'])

    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print('RMSE -',np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print("---------------------")
    print("Score - ",metrics.r2_score(y_test,predictions))

    params['algorithms']["svm"] = (metrics.r2_score(y_test, predictions)) * 100

    params["algokeys"] = list(params["algorithms"].keys())
    params["algovalues"] = list(params["algorithms"].values())

    if params["best_acc"] < round(params['algorithms']["svm"], 2):
        params["best_acc"] = round(params['algorithms']["svm"], 2)
    
