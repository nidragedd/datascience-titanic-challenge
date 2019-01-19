"""
Created on 19/01/2019
@author: nidragedd

XGBoost implementation for this specific problem
"""
from xgboost import XGBClassifier

from data.data_transformation import specific_normalization
from models.helper import grid_search_tuning, split_data, fit_and_score


def grid_search(X_train, Y_train):
    """
    Build a GridSearchCV for XGBoost classifier and display best tuning parameters to use them later
    :param X_train: (pandas Dataframe) the X split of 'Training' dataset (i.e all features)
    :param Y_train: (pandas Dataframe) the Y split of 'Training' dataset (i.e the target)
    :return: XGBoost best classifier found
    """
    params = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'n_estimators': [1000, 2000, 3000],
        'reg_alpha': [0.01, 0.02, 0.03, 0.04]
    }

    clf = XGBClassifier(random_state=42, learning_rate=0.05)
    return grid_search_tuning(params, clf, X_train, Y_train)


def build_and_train_xgb(train_df, use_grid_search=False):
    """
    Build XGBoost classifier model.
    :param train_df: (pandas Dataframe) the whole 'Training' dataset
    :param use_grid_search: (boolean) if True we will used GridSearch to look for best parameters, if False (by default)
    we assume that this has already been done and we use some parameters found from a previous run --> the program runs
    much faster
    :return: XGBoost classifier as the ML model
    """
    train_df = specific_normalization(train_df)
    X_train, Y_train, X_train_split, Y_train_split, X_test_split, Y_test_split = split_data(train_df)

    model = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    if use_grid_search:
        model = grid_search(X_train, Y_train)

    fit_and_score('XGBoost', model, X_train, Y_train, X_test_split, Y_test_split)

    return model
