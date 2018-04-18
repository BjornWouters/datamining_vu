#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet

def import_data(ID):
    df = pd.read_csv(ID, index_col=0)
    return df

def split_test_set(sample_data):
    test_set = sample_data[-1:]
    training_set = sample_data[0:-1]
    #print(test_set, training_set)
    
    return test_set, training_set
    
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
   elastic_model = elasticnet(X,y)
   recurrent_model = recurrent_neural_network(X,y)

def benchmark(test_set, training_set, y_prediction, y_true):
    y_true.append(test_set.mood[-1])
    y_prediction.append(training_set[-1:].mood[-1])
    return y_prediction, y_true


def elasticnet(X,y):
    regr = ElasticNet(random_state=0)   
    regr.fit(X, y)
    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=False,random_state=0, selection='cyclic', tol=0.0001, warm_start=False)
    print(regr.coef_, regr.intercept_)
    
    return #model?

def recurrent_neural_network(X,y):
    #
    # Multi-layer Perceptron implementeren?
    #
    return #model?

#main program 
os.chdir( '/Users/amber/Desktop/data_mining' )
filenames = glob.glob( '*/**.csv' )
benchmark_y_prediction = []
benchmark_y_true = []

for sample_id in filenames:
    sample_data = import_data(sample_id)
    test_set, training_set = split_test_set(sample_data)
    benchmark_y_prediction, benchmark_y_true = benchmark(test_set, training_set, benchmark_y_prediction, benchmark_y_true)
    #split_data(training_set) #voor de timeseriessplit, zodat daarna de modellen gemaakt kunnen worden.
#
#modellen testen en AUC uitrekenen, voor alle modellen (voor benchmark dus y_prediction met y_test vergelijken)
#
