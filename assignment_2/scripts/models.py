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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


def import_data():
    training_set = pd.read_csv("../results/prepared_train.csv", sep=',')
    original_dataset = pd.read_csv("../results/predict_dataset.csv", sep=',')
    # training_set = pd.read_csv("prep_small_train.csv", sep=',')
    # test_set = pd.read_csv("prep_small_train.txt", sep='\t')
    dropped_features = ['prop_starrating', 'click_bool', 'booking_bool']
    X = training_set.drop(dropped_features, axis=1)
    y = training_set.booking_bool
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    global N_FEATURES
    N_FEATURES = len(X.columns)

    return X_train, y_train, X_test, original_dataset


def benchmark():
    return


"""
def GBM(X_train, X_validation, y_train, y_validation):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X_validation, y_validation) 
    print(clf.feature_importances_)
    return 
"""


def RF(X, y):
    global N_FEATURES
    clf = RandomForestRegressor(max_depth=2, min_samples_split=2, random_state=None,
                                 max_features='sqrt', n_jobs=1)
    clf.fit(X, y)
    # print(clf.feature_importances_)

    return clf


def lambdamart(X, y):
    clf = XGBClassifier(rank='ndcg')
    clf.fit(X, y)

    return clf


def predictions(original_dataset, test_set, clfrf):
    srch_ids = original_dataset.srch_id.unique()
    predictionsrf = list()
    predictionslm = list()
    used_properties = list()
    for i, id in enumerate(srch_ids):
        listings = original_dataset[original_dataset.srch_id == id]
        test_row = test_set[test_set.prop_id.isin(list(listings.prop_id))]
        used_properties.append(test_row.prop_id)
        test_row = test_row.drop('prop_id', axis=1)

        if len(test_row) == 0:
            continue

        outputrf = clfrf.predict(test_row)
        # outputlm = clflm.predict(test_row)
        predictionsrf.append(outputrf)
        # predictionslm.append(outputlm)
        if i == 2000:
            break
    nDCG(predictionsrf, used_properties, original_dataset)

    return


def nDCG(predictionsrf, used_properties, original_dataset):
    random_score_list = list()
    final_score_list = list()
    srch_ids = original_dataset.srch_id.unique()
    for i, id in enumerate(srch_ids):
        listings = original_dataset[original_dataset.srch_id == id]
        listings = listings[listings.prop_id.isin(list(used_properties[i]))]
        predicted_index = np.argsort(predictionsrf[i])
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

    print('Final score: {}'.format(np.mean(final_score_list)))
    print('Final score (randomized): {}'.format(np.mean(random_score_list)))

    # matrix = pd.concat([click_and_booking_bool, prediction])
    # print(matrix)
    # matrix srch_id, prop_id, score (clicked=1, booked=5), discounted score (score/position)
    # matrix["discounted_score"] = matrix.
    # take sum of discounted score
    # normalize the score by dividing the total for this answer by the maximum for this query
    # creating the best answer is the properties sorted first by booking_boolean and then click_boolean
    # final nDCG = our score / best score
    # take the average of all the predicted srch_id's

    return


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
# benchmark()
# GBM(X,y)
clfrf = RF(X_train.drop('prop_id', axis=1), y_train)
# clflm = lambdamart(X, y)
predictions(original_dataset, X_test, clfrf)
