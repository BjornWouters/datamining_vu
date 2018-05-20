#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def import_data():
    training_set = pd.read_csv("prepared_train.csv", sep=',', nrows=1000)
    original_dataset = pd.read_csv("predict_dataset.csv", sep=',', nrows=100)
    #training_set = pd.read_csv("prep_small_train.csv", sep=',')
    #test_set = pd.read_csv("prep_small_train.txt", sep='\t') 
    X = training_set.loc[:, "prop_starrating"::]
    y = training_set.prop_id
    global N_FEATURES
    N_FEATURES = len(X.columns)

    return X, y, original_dataset
    
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

def RF(X, y, original_dataset):
    global N_FEATURES
    clf = RandomForestClassifier(max_depth=2, min_samples_split=2, random_state=None, max_features='sqrt', n_jobs=1)
    clf.fit(X, y)
    #print(clf.feature_importances_)
    
    return clf
 
def lambdamart(X, y):
    clf = XGBClassifier(rank='ndcg')
    clf.fit(X, y)
    
    return clf

def predictions(original_dataset, clfrf, clflm):
    srch_ids = original_dataset.srch_id.unique()
    predictionsrf = []
    predictionslm = []
    for id in srch_ids:
        listings = original_dataset[original_dataset.srch_id == id]
        outputrf = clfrf.predict_proba(listings.loc[:, "prop_starrating"::])
        outputlm = clflm.predict_proba(listings.loc[:, "prop_starrating"::])
        predictionsrf.append(outputrf)
        predictionslm.append(outputlm)
    nDCG(predictionsrf, predictionslm)
    
    return
 
def nDCG(predictionsrf, predictionslm):
    max_df = max_scores(original_dataset)
    print(max_df)
    
    
    #matrix = pd.concat([click_and_booking_bool, prediction])
    #print(matrix)
    #matrix srch_id, prop_id, score (clicked=1, booked=5), discounted score (score/position)
    #matrix["discounted_score"] = matrix. 
    #take sum of discounted score
    #normalize the score by dividing the total for this answer by the maximum for this query
    #creating the best answer is the properties sorted first by booking_boolean and then click_boolean
    #final nDCG = our score / best score
    #take the average of all the predicted srch_id's
    
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
            discount_score = score/i

        sum_score += discount_score

    return sum_score


def max_scores(df):
    score_list = list()
    id_list = list()
    srch_ids = df.srch_id.unique()
    for id in srch_ids:
        listings = df[df.srch_id == id]
        sorted_listings = listings.sort_values(by=['booking_bool', 'click_bool'], ascending=False)
        score = calculate_score(sorted_listings)

        id_list.append(id)
        score_list.append(score)

    max_df = pd.DataFrame(data={'listing_id': id_list, 'score': score_list})
    return max_df

def submission_file():
    with open("predictions.csv", "w+") as f:
        f.write("\n".join("blabla"))
    
# main program
X, y, original_dataset = import_data()
#benchmark()
#GBM(X,y)
clfrf = RF(X,y, original_dataset)
clflm = lambdamart(X,y)
predictions(original_dataset, clfrf, clflm)
