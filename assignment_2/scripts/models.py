#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor



def import_data():
    training_set = pd.read_csv("../results/prepared_train.csv", sep=',')
    original_dataset = pd.read_csv("../results/predict_dataset.csv", sep=',')
    dropped_features = ['prop_starrating', 'position', 'click_bool', 'booking_bool']
    X = training_set.drop(dropped_features, axis=1).fillna(0)
    y = training_set.click_bool
    X_train, X_validation, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    test_set = pd.read_csv("../results/test_set.csv", sep=',').fillna(0)
    test_set = test_set.drop(['prop_starrating'], axis=1)
    return X_train, y_train, X_validation, original_dataset, test_set


def GBM(X, y):
    clf = GradientBoostingRegressor(max_features="log2", random_state=8, n_estimators=1500, max_depth=2,
                                    min_samples_split=2, min_samples_leaf=1, learning_rate=0.01,
                                    loss='quantile', alpha=0.7)
    clf.fit(X, y)
    joblib.dump(clf, '../results/GBM_model.pkl') 
    print('Feature importances GBM: {}'.format(clf.feature_importances_), X.columns.values)
    plot_feature_importances(clf.feature_importances_, X.columns.values)
    print(clf)
    return clf


def RF(X, y):
    clf = RandomForestRegressor(n_estimators=20, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                random_state=4, max_features='log2', n_jobs=1)
    clf.fit(X, y)
    joblib.dump(clf, '../results/RF_model.pkl') 
    print('Feature importances RF: {}'.format(clf.feature_importances_))
    plot_feature_importances(clf.feature_importances_, X.columns.values)
    print(clf)
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

    rf = 'RF'
    gbm = 'GBM'
    nDCG_RF = nDCG(predictionsrf, used_properties, original_dataset, rf)
    nDCG_GBM = nDCG(predictionsgbm, used_properties, original_dataset, gbm)

    return nDCG_RF, nDCG_GBM


def nDCG(predictions, used_properties, original_dataset, method):
    random_score_list = list()
    final_score_list = list()
    srch_ids = original_dataset.srch_id.unique()
    with open('../results/test_output.txt', 'w') as test_output:
        for i, id in enumerate(srch_ids):
            listings = original_dataset[original_dataset.srch_id == id]
            listings = listings[listings.prop_id.isin(list(used_properties[i]))]
            predicted_index = np.argsort(predictions[i])
            if len(listings) != len(predicted_index):
                break
            listings['predicted_index'] = predicted_index
            score = calculate_score(listings.sort_values('predicted_index'), id, test_output)
            if score == 0:
                final_score = 0
            else:
                max_score = calculate_score(listings, None, None)
                final_score = score / max_score
            final_score_list.append(final_score)
            # Calculate random shuffle scores
            listings = listings.sample(frac=1)
            score = calculate_score(listings, None, None)
            if score == 0:
                final_score = 0
            else:
                max_score = calculate_score(listings, None, None)
                final_score = score / max_score
            random_score_list.append(final_score)

    mean_final_scores = np.mean(final_score_list)
    print('Final score {}: {}'.format(method, mean_final_scores))
    print('Final score (randomized): {}'.format(np.mean(random_score_list)))

    return mean_final_scores


def calculate_score(listings, id, test_output):
    sum_score = 0
    for i, row in enumerate(listings.iterrows(), start=1):
        index, listing = row
        if test_output:
            test_output.write(str(int(listing.srch_id)) + ', ' + str(int(listing.prop_id)) + '\n')
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
        score = calculate_score(sorted_listings, None, None)

        id_list.append(id)
        score_list.append(score)
        if i == 5:
            break
    max_df = pd.DataFrame(data={'srch_id': id_list, 'score': score_list})
    
    return max_df


def predict_test_set(clf, test_set):
    srch_ids = test_set.srch_id.unique()
    dropped_features = ['srch_id', 'prop_id']
    with open('../results/submission.txt', 'w') as submission:
        submission.write('SearchId,PropertyId' + '\n')
        for i, id in enumerate(srch_ids):
            listings = test_set[test_set.srch_id == id]
            output_test_set = clf.predict(listings.drop(dropped_features, axis=1))
            predicted_index = np.argsort(output_test_set)
            listings['predicted_index'] = predicted_index
            listings = listings.sort_values('predicted_index')
            for i, row in enumerate(listings.iterrows(), start=1):
                index, listing = row
                submission.write(
                    str(int(listing.srch_id)) + ',' + str(int(listing.prop_id)) + '\n')

def plot_feature_importances(importances, values):
    feature_importance = 100.0 * (importances / importances.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, values[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()
        
    
def main():
    # Set useless warning to None
    pd.options.mode.chained_assignment = None

    # main program
    X_train, y_train, X_validation, original_dataset, test_set = import_data()
    clfgbm = GBM(X_train.drop('prop_id', axis=1), y_train)
    clfrf = RF(X_train.drop('prop_id', axis=1), y_train)
    nDCG_RF, nDCG_GBM = predictions(original_dataset, X_validation, clfrf, clfgbm)

    if nDCG_RF > nDCG_GBM:
        predict_test_set(clfrf, test_set)
    else:
        predict_test_set(clfgbm, test_set)
    #predict test set with the best model


if __name__ == '__main__':
    main()
