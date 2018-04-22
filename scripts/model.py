#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_curve
 
def import_data(ID):
    df = pd.read_csv(ID, index_col=0)
    return df

def split_test_set(sample_data):
    test_set = sample_data[-1:]
    training_set = sample_data[0:-1]
    #print(test_set, training_set)
    
    return test_set, training_set
    
def split_data(sample_data, test_set):
   #split in test and trainingset and timeseries cross validation
   X = sample_data.loc[:,"activity"::]
   y = sample_data.mood
   X_test = test_set.loc[:,"activity"::]
   y_test = test_set.mood
   
   tscv = TimeSeriesSplit(max_train_size=None, n_splits=11)
   #NOG BEKIJKEN, raise KeyError('%s not in index' % objarr[mask]) KeyError: '[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13] not in index'
   #for train_index, test_index in tscv.split(X):
   #    print("TRAIN:", train_index, "TEST:", test_index)
   #    X_train, X_test = X[train_index], X[test_index]
   #    y_train, y_test = y[train_index], y[test_index]
       
   #Nog bekijken --> NaN pikt 'ie niet, alle NaN veranderen naar 0 kan lijkt mij niet overal?
   
   return X, y, X_test, y_test

def benchmark(test_set, training_set, y_prediction, y_true):
    y_true.append(test_set.mood[-1])
    y_prediction.append(training_set[-1:].mood[-1])
    return y_prediction, y_true

def elasticnet(X,y,X_test): #heb nu even cv gedaan in de elasticnet, kunnen we veradenren als timeseriesspplit werkt (=3x CV verwijderen)
    regr = ElasticNetCV(cv=5, random_state=0)   
    y_prediction_elasticnet = regr.fit(X, y).predict(X_test)
    #fpr, tpr = roc_curve(y_test, y_prediction_elasticnet, sample_weight=None, drop_intermediate=True)
    return y_prediction_elasticnet

def recurrent_neural_network(X,y, X_test):
    regr = MLPRegressor(hidden_layer_sizes=(100,))
    regr.fit(X,y)
    y_prediction_nn = regr.predict(X_test)
    return y_prediction_nn

#main program 
#find all files with prepared data
os.chdir( '/Users/amber/Desktop/data_mining/results' )
filenames = glob.glob( '*/**.csv' )
benchmark_y_prediction = []
benchmark_y_true = []
elasticnet_y_prediction = []
recurrentnn_y_prediction = []

for sample_id in filenames:
    sample_data = import_data(sample_id)
    test_set, training_set = split_test_set(sample_data)
    benchmark_y_prediction, benchmark_y_true = benchmark(test_set, training_set, benchmark_y_prediction, benchmark_y_true)
    X, y, X_test, y_test = split_data(training_set, test_set) 
    elasticnet_y_prediction.append(elasticnet(X,y,X_test))
    recurrentnn_y_prediction.append(recurrent_neural_network(X,y, X_test))


print(benchmark_y_prediction, elasticnet_y_prediction, recurrentnn_y_prediction)
