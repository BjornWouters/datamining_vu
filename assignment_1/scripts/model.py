#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from numpy import nan as Nan
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_curve, mean_squared_error


def import_data(ID):
    df = pd.read_csv(ID, index_col=0)
    return df


def split_test_set(sample_data):
    test_set = sample_data[-1:]
    training_set = sample_data[0:-1]
    # print(test_set, training_set)

    return test_set, training_set


def split_data(sample_data, test_set):
    sample_data = sample_data.reset_index()
    # split in test and trainingset and timeseries cross validation
    X = sample_data.loc[:, "activity"::]
    y = sample_data.mood
    X_test = test_set.loc[:, "activity"::]
    y_test = test_set.mood

    tscv = TimeSeriesSplit(n_splits=25)
    return X, y, tscv.split(X)

    # Nog bekijken --> NaN pikt 'ie niet, alle NaN veranderen naar 0 kan lijkt mij niet overal?


def benchmark(train, test):
    RMSE = math.sqrt(math.pow(list(train)[-1] - float(list(test)[-1]), 2))
    return RMSE


def elasticnet(X, y, test_set,
               validation):  # heb nu even cv gedaan in de elasticnet, kunnen we veradenren als timeseriesspplit werkt (=3x CV verwijderen)
    regr = ElasticNet(random_state=0)
    X = X.fillna(0)
    test_set = test_set.fillna(0)

    regr.fit(X, y)
    y_prediction_elasticnet = regr.predict(test_set)
    # fpr, tpr = roc_curve(y_test, y_prediction_elasticnet, sample_weight=None, drop_intermediate=True)
    RMSE = math.sqrt(mean_squared_error(y_prediction_elasticnet, validation))

    return RMSE


def recurrent_neural_network(X, y, test_set, validation):
    regr = MLPRegressor(hidden_layer_sizes=(500,), solver='lbfgs', activation='relu')
    X = X.fillna(0)
    test_set = test_set.fillna(0)

    regr.fit(X, y)
    y_prediction_nn = regr.predict(test_set)
    RMSE = math.sqrt(mean_squared_error(y_prediction_nn, validation))
    return RMSE


# main program
# find all files with prepared data
os.chdir('../results')
filenames = glob.glob('*/**.csv')
results = dict()
result_df = pd.DataFrame()

for i, sample_id in enumerate(filenames):
    sample_name = sample_id.split('/')[-1].strip('.csv')
    if sample_name not in results:
        results.update({sample_name: {
            'EN': list(),
            'MLP': list(),
            'bench': list()
        }})
        serie = pd.Series([Nan, Nan, Nan], index=['EN', 'MLP', 'bench'])
        serie.name = sample_name
        result_df = result_df.append(serie)

    benchmark_y_true = []

    sample_data = import_data(sample_id)
    test_set, training_set = split_test_set(sample_data)
    X, y, split_iter = split_data(training_set, test_set)
    for train_index, test_index in split_iter:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
        y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]

        if 'normalized_normal' in sample_id:
            results[sample_name]['EN'].append(elasticnet(X_train, y_train, X_test, y_test))
            results[sample_name]['bench'].append(benchmark(y_train, y_test))

        if 'recurrend' in sample_id:
            results[sample_name]['MLP'].append(recurrent_neural_network(X_train, y_train, X_test, y_test))


for key in results.keys():
    sample_results = results[key]
    print('Sample {}:\nelasticnet: {} \nMLP {} \nbenchmark {}\n\n'.format(key,
                                                                          str(np.mean(sample_results['EN'])),
                                                                          str(np.mean(sample_results['MLP'])),
                                                                          str(np.mean(sample_results['bench']))))
    result_df.loc[key]['EN'] = np.mean(sample_results['EN'])
    result_df.loc[key]['bench'] = np.mean(sample_results['bench'])
    result_df.loc[key]['MLP'] = np.mean(sample_results['MLP'])

print('Mean EN: ' + str(result_df.EN.mean()))
print('Mean benchmark: ' + str(result_df.bench.mean()))
print('Mean MLP: ' + str(result_df.MLP.mean()))

result_df.plot()
plt.xticks(range(len(results.keys())), range(len(results.keys())))
plt.title('Score differences per model')
plt.ylabel('Average root mean squared error')
plt.xlabel('Patient')
plt.show()