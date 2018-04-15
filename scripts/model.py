#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def import_data(ID):
    df = pd.read_csv(ID, index_col=0)
    print(df)
    return df

def split_data(sample_data):
   #split in test and trainingset and timeseries cross validation
   X = sample_data.loc[:,"activity"::]
   y = sample_data.mood
   tscv = TimeSeriesSplit(max_train_size=None, n_splits=11)
   
   #NOG BEKIJKEN, raise KeyError('%s not in index' % objarr[mask]) KeyError: '[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13] not in index'
   """for train_index, test_index in tscv.split(X):
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]"""
       
   #Nog bekijken --> NaN pikt 'ie niet, alle NaN veranderen naar 0 kan lijkt mij niet overal?
   #elasticnet(X,y)
   gradient_boosting_regression(X,y)


def elasticnet(X,y):
    regr = ElasticNet(random_state=0)   
    regr.fit(X, y)
    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=False,random_state=0, selection='cyclic', tol=0.0001, warm_start=False)
    print(regr.coef_, regr.intercept_)

def gradient_boosting_regression(X,y):
    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X, y)
    clf.fit(X, y)
    mse = mean_squared_error(y, clf.predict(X))
    print("MSE: %.4f" % mse)
    

#main program 
os.chdir( '/Users/amber/Desktop/data_mining' )
filenames = glob.glob( '*/**.csv' )
for sample_id in filenames:
    sample_data = import_data(sample_id)
    split_data(sample_data)


