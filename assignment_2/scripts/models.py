#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def import_data():
    training_set = pd.read_csv("prepared_train.csv", sep=',', nrows=1000)
    original_dataset = pd.read_csv("training_set_VU_DM_2014.csv, sep=',', nrows=1000)
    predict_set = original_dataset.loc[:,"srch_id":"prop_id":]
    #training_set = pd.read_csv("prep_small_train.csv", sep=',')
    #test_set = pd.read_csv("prep_small_train.txt", sep='\t') 
    X = training_set.loc[:, "prop_starrating":"promotion_flag":]
    click_and_booking_bool = training_set.loc[:,"click_bool":"booking_bool":]
    y = training_set.prop_id
    global N_FEATURES
    N_FEATURES = len(X.columns)

    return X, y, click_and_booking_bool

def split_validation_set(X, y):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)
    #benchmark()
    GBM(X_train, X_validation, y_train, y_validation)
    #RF(X_train, X_validation, y_train, y_validation)
    #lambdamart(X_train, X_validation, y_train, y_validation)
    
def benchmark():
    return

def GBM(X_train, X_validation, y_train, y_validation):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X_validation, y_validation) 
    print(clf.feature_importances_)
    #mean_accuracy = clf.score(X_validation, y_validation)
    return 

def RF(X_train, X_validation, y_train, y_validation):
    global N_FEATURES
    clf = RandomForestClassifier(max_depth=2, min_samples_split=2, random_state=None, max_features='sqrt', n_jobs=1)
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
    y_prediction = clf.predict(X_validation)
    output = clf.predict_proba(X_validation)
    print(accuracy_score(y_validation, y_prediction))
    print(len(y_validation), len(y_prediction))
    output = pd.DataFrame(output, columns=y_train)
    nDCG(output)
    

    return
 
def lambdamart(X_train, X_validation, y_train, y_validation):
    clf = XGBClassifier(rank='ndcg')
    clf.fit(X_train, y_train)
    print(clf.score(X_validation, y_validation))
    print(clf.feature_importances_)
    mean_accuracy = clf.score(X_validation, y_validation)
    
    return
    
def nDCG(prediction):
    matrix = pd.concat([click_and_booking_bool, prediction])
    print(matrix)
    #matrix srch_id, prop_id, score (clicked=1, booked=5), discounted score (score/position)
    #matrix["discounted_score"] = matrix. 
    #take sum of discounted score
    #normalize the score by dividing the total for this answer by the maximum for this query
    #creating the best answer is the properties sorted first by booking_boolean and then click_boolean
    #final nDCG = our score / best score
    #take the average of all the predicted srch_id's
    
    return

def submission_file():
    with open("predictions.csv", "w+") as f:
        f.write("\n".join("blabla"))
    
# main program
X, y, click_and_booking_bool= import_data()
split_validation_set(X, y)
