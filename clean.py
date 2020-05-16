#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:09:54 2020

@author: shantanu
"""

import pandas as pd
import numpy as np
from sklearn.impute._iterative import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor


def cleaning(data,y,params):
    size = len(data)
    ids = ["ID","id","Id","iD"]
    def isId(i):
        for j in ids:
            if j in i:
                return True
        return False
    print("="*100)
    print("Selecting unnecessary columns for dropping...")
    drp = set()
    for i in data.columns:
        if (isId(i)) or (data[i].dtype == 'O' and len(data[i].value_counts())>300):
            drp.add(i)

    params["nullcount"] = sum(data.isnull().sum()>0)

    for i in data.columns:
        if data[i].isnull().sum()>=size//2:
            drp.add(i)
    
    data.drop(drp,axis=1,inplace = True)
    print("Dropped Columns - ",drp)
    
    print("="*100)
    
    print("Selecting Categorical and Numerical columns...")
    cat_col = []
    num_col = []
    for i in data.columns:
        if data[i].dtype == 'O':
            cat_col.append(i)
    
    num_col = list(data.select_dtypes(include=np.number).columns)
    print("Categorical columns are: ",cat_col)
    print("Numerical Columns are: ",num_col)
    
    print("="*100)
    
    print("Filling NULL values...")
    for i in cat_col:
        data[i] = data[i].fillna("Missing")

    imp = IterativeImputer(random_state=123)
    data[num_col] = pd.DataFrame(imp.fit_transform(data[num_col]), columns=num_col)
    print("NULL Values removed!")
    print("Checking is any null values are left: ",any(data.isnull().sum()>0))
    
    print("="*100)
    
    if len(cat_col)!=0:
        for i in cat_col:
            temp = data.groupby(i).size() / size
            data.loc[:, i + '_val'] = data[i].map(temp)
            data.loc[data[i + '_val'] <= 0.01, i] = 'Rare'
            data.drop(i + '_val', axis=1, inplace=True)

        print("Creating dummy variables...")
        data = pd.get_dummies(data,drop_first=True)


    print("Final shape of dataset: ",data.shape)
    
    print("="*100)
    
    print("Dropping Highly correlated features...")    
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.80)]
    data.drop(to_drop,axis=1,inplace=True)
    print("Features Dropped: ",to_drop)
    
    print("="*100)
    
    print("Selecting important Features...")    
    mms = MinMaxScaler()
    data_scaled = mms.fit_transform(data)
    selector1 = LassoCV()
    selector1 = SelectFromModel(selector1)
    selector1.fit(data_scaled,y)
    selector1.get_support()
    feat_selected1 = list(data.columns[(selector1.get_support())])

    feat_imp = set()
    temp1 = list((selector1.estimator_.coef_)[(selector1.get_support())])
    mx = max(temp1)
    mn = min(temp1)

    def transform(x):
        return (x - mn) / (mx - mn)

    for i in range(len(temp1)):
        feat_imp.add((feat_selected1[i], round(transform(abs(temp1[i]))*100,2)))
    feat_imp = list(feat_imp)
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    feat_show = feat_imp[:min(5, len(feat_imp))]
    feat_show[-1] = (feat_show[-1][0], feat_show[-1][1]+1)
    params["features"] = feat_show

    print(params["features"])

    selector2 = RandomForestRegressor()
    selector2 = SelectFromModel(selector2)
    selector2.fit(data_scaled,y)
    selector2.get_support()
    feat_selected2 = list(data.columns[(selector2.get_support())])
    feat_selected = list(set(feat_selected1+feat_selected2))
    
    print("Found {} important features".format(len(feat_selected)))

    params["feat_count"] = len(feat_selected)

    data_feat = data[feat_selected]
    print("final shape of dataset",data_feat.shape)
    
    print("="*100)
    
    return data_feat,y,params