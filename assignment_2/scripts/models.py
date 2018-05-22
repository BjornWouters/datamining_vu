#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor



def import_data():
    training_set = pd.read_csv("../results/prepared_train.csv", sep=',')
    original_dataset = pd.read_csv("../results/predict_dataset.csv", sep=',')
    dropped_features = ['prop_starrating', 'click_bool', 'booking_bool']
    X = training_set.drop(dropped_features, axis=1)
    y = training_set.booking_bool
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, y_train, X_test, original_dataset


def GBM(X, y):
    clf = GradientBoostingRegressor(random_state=None, n_estimators=1500, max_depth=2, min_samples_split=2, learning_rate=0.01, loss='quantile', alpha=0.7)
    clf.fit(X, y)
    print('Feature importances GBM: {}'.format(clf.feature_importances_))
    
    return clf


def RF(X, y):
    clf = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=2, random_state=4, max_features='sqrt', n_jobs=1)
    clf.fit(X, y)
    print('Feature importances RF: {}'.format(clf.feature_importances_))
    
    return clf


def predictions(original_dataset, test_set, clfrf, clfgbm):
    srch_ids = original_dataset.srch_id.unique()
    predictionsrf = list()
    predictionsgbm = list()
    used_properties = list()
    for i, id in enumerate(srch_ids):
        listings = original_dataset[original_dataset.srch_id == id]
        test_row = test_set[test_set.prop_id.isin(list(listings.prop_id))]
        used_properties.append(test_row.prop_id)
        test_row = test_row.drop('prop_id', axis=1)
        if len(test_row) == 0:
            continue

        outputrf = clfrf.predict(test_row)
        outputgbm = clfgbm.predict(test_row)
        predictionsrf.append(outputrf)
        predictionsgbm.append(outputgbm)
        if i == 2000:
            break
    print(predictionsrf)
    rf='RF'
    gbm='GBM'
    nDCG_RF = nDCG(predictionsrf, used_properties, original_dataset, rf)
    nDCG_GBM = nDCG(predictionsgbm, used_properties, original_dataset, gbm)

    return nDCG_RF, nDCG_GBM


def nDCG(predictions, used_properties, original_dataset, method):
    random_score_list = list()
    final_score_list = list()
    srch_ids = original_dataset.srch_id.unique()
    for i, id in enumerate(srch_ids):
        listings = original_dataset[original_dataset.srch_id == id]
        listings = listings[listings.prop_id.isin(list(used_properties[i]))]
        predicted_index = np.argsort(predictions[i])
        if len(listings) != len(predicted_index):
            break
        listings['predicted_index'] = predicted_index
        score = calculate_score(listings.sort_values('predicted_index'))
        if score == 0:
            final_score = 0
        else:
            max_score = calculate_score(listings)
            final_score = score / max_score
        final_score_list.append(final_score)

        # Calculate random shuffle scores
        listings = listings.sample(frac=1)
        score = calculate_score(listings)
        if score == 0:
            final_score = 0
        else:
            max_score = calculate_score(listings)
            final_score = score / max_score
        random_score_list.append(final_score)

        if i == 2000:
            break

    mean_final_scores = np.mean(final_score_list)
    print('Final score {}: {}'.format(method, mean_final_scores))
    print('Final score (randomized): {}'.format(np.mean(random_score_list)))

    return mean_final_scores


def calculate_score(listings):
    sum_score = 0
    for i, row in enumerate(listings.iterrows(), start=1):
        index, listing = row
        score = 0
        if listing.booking_bool == 1:
            score = 5
        elif listing.click_bool == 1:
            score = 1

        if score == 0:
            discount_score = 0
        else:
            discount_score = score / i

        sum_score += discount_score

    return sum_score


def max_scores(df):
    score_list = list()
    id_list = list()
    srch_ids = df.srch_id.unique()
    for i, id in enumerate(srch_ids):
        listings = df[df.srch_id == id]
        sorted_listings = listings.sort_values(by=['booking_bool', 'click_bool'], ascending=False)
        score = calculate_score(sorted_listings)

        id_list.append(id)
        score_list.append(score)
        if i == 5:
            break
    max_df = pd.DataFrame(data={'srch_id': id_list, 'score': score_list})
    
    return max_df


def submission_file():
    with open("predictions.csv", "w+") as f:
        f.write("\n".join("blabla"))


# main program
X_train, y_train, X_test, original_dataset = import_data()
clfgbm = GBM(X_train.drop('prop_id', axis=1), y_train)
clfrf = RF(X_train.drop('prop_id', axis=1), y_train)
nDCG_RF, nDCG_GBM = predictions(original_dataset, X_test, clfrf, clfgbm)

#if nDCG_RF > nDCG_GBM:
#    submission_file(nDCG_RF)
#else:
#    submission_file(nDCG_GBM)
