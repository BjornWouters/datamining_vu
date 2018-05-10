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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def import_data():
    #training_set = pd.read_csv(ID, index_col=0)
    #test_set = pd.read_csv(ID, index_col=0)   
    #X =
    #y =
    #X_test =
    #y_test =
    return X_train, y_train, X_test, y_test

def split_validation_set(X, y):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)
    benchmark()
    GBM(X_train, X_validation, y_train, y_validation)
    RF(X_train, X_validation, y_train, y_validation)
    lambdamart(X_train, X_validation, y_train, y_validation)
    
def benchmark():
    return

def GBM(X_train, X_validation, y_train, y_validation):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test) 
    clf.feature_importances_
    mean_accuracy = clf.score(X_test, y_test)
    return 

def RF(X_train, X_validation, y_train, y_validation):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
    mean_accuracy = clf.score(X_test, y_test)
    return
 
def lambdamart(X_train, X_validation, y_train, y_validation):
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test) 
    clf.feature_importances_
    mean_accuracy = clf.score(X_test, y_test)
    return
    
def nDCG():
    return

def submission_file():
    with open("predictions.csv", "w+") as f:
        f.write("\n".join("blabla"))
    
# main program
X, y, X_test, y_test = import_data()
split_validation_set(X, y)
