"""
Created on 14/01/2019
@author: nidragedd

RandomForest implementation for this specific problem
"""
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from data.data_transformation import specific_normalization
from models.helper import grid_search_tuning, split_data, fit_and_score


def grid_search(X_train, Y_train):
    """
    Build a GridSearchCV for RandomForest classifier and display best tuning parameters to use them later
    :param X_train: (pandas Dataframe) the X split of 'Training' dataset (i.e all features)
    :param Y_train: (pandas Dataframe) the Y split of 'Training' dataset (i.e the target)
    :return: sklearn RandomForest best classifier found
    """
    params = {
        'n_estimators': range(100, 1000, 100),
        'max_features': [0.5, 1.],
        'max_depth': [5., None]
    }
    clf = RandomForestClassifier(random_state=42)
    return grid_search_tuning(params, clf, X_train, Y_train)


def build_and_train_rf(train_df, use_grid_search=False):
    """
    Build a RandomForest classifier model. Parameters have been found with the help of GridSearchCV.
    Otherwise, setting the last argument to True will build a new classifier using Hyper-Parameter Tuning
    :param train_df: (pandas Dataframe) the whole 'Training' dataset
    :param use_grid_search: (boolean) if True we will used GridSearch to look for best parameters, if False (by default)
    we assume that this has already been done and we use some parameters found from a previous run --> the program runs
    much faster
    :return: sklearn RandomForest classifier as the ML model
    """
    model = RandomForestClassifier(n_estimators=900, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                   max_features=0.5)
    X_train, Y_train, X_train_split, Y_train_split, X_test_split, Y_test_split = split_data(train_df)

    if use_grid_search:
        model = grid_search(X_train, Y_train)

    fit_and_score('Random Forest', model, X_train, Y_train, X_test_split, Y_test_split)

    return model
