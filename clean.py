#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:09:54 2020

@author: shantanu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor


def cleaning(data,y):
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
        m = data[i].mode()
        data[i] = data[i].fillna(m)
        
    for i in num_col:
        m = data[i].mean()
        data[i] = data[i].fillna(m)
    print("NULL Values removed!")
    print("Checkiing is any null values are left: ",any(data.isnull().sum()>0))
    
    print("="*100)
    
    if len(cat_col)!=0:
        print("Creating dummy variables...")
        cat_features = pd.get_dummies(data[cat_col],drop_first=True)
        print("Final shape of categorical features: ",cat_features.shape)
    
    if len(num_col)!=0:
        num_features = data[num_col]
        print("Final shape of numerical features: ",num_features.shape)
    
    if len(cat_col)==0:
        data = num_features
    elif len(num_col) == 0:
        data = cat_features
    else:
        data = pd.concat([num_features,cat_features],axis=1)
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
    
    selector2 = RandomForestRegressor()
    selector2 = SelectFromModel(selector2)
    selector2.fit(data_scaled,y)
    selector2.get_support()
    feat_selected2 = list(data.columns[(selector2.get_support())])
    feat_selected = list(set(feat_selected1+feat_selected2))
    
    print("Found {} important features".format(len(feat_selected)))
    
    data_feat = data[feat_selected]
    print("final shape of dataset",data_feat.shape)
    
    print("="*100)
    
    return data_feat,y