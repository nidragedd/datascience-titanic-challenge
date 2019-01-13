"""
Created on 22/12/2018
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from data_preparation import build_fare_means, build_age_table, handle_missing_values
from data_transformation import transform_and_create_new_features
from load_and_save import build_and_save_from_output


def grid_search_for_rf(train_df):
    """
    Build a GridSearch Cross Validation for RandomForest classifier and display best tuning parameters to use them later
    :param train_df: (pandas Dataframe) the whole 'Training' dataset
    """
    # Hyper-parameters management
    hyperparams = {
        'n_estimators': range(100, 1000, 100),
        'max_features': [0.5, 1.],
        'max_depth': [5., None]
    }
    X = train_df.drop(['Survived'], axis=1)
    Y = train_df['Survived']
    gs = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=hyperparams, verbose=True, cv=5)
    gs.fit(X, Y)
    print("Grid search best score is {}".format(gs.best_score_))
    print("Grid search best estimator is {}".format(gs.best_estimator_))
    # Will print:
    # Grid search best score is 0.797979797979798
    # Grid search best estimator is RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #             max_depth=5.0, max_features=0.5, max_leaf_nodes=None,
    #             min_impurity_decrease=0.0, min_impurity_split=None,
    #             min_samples_leaf=1, min_samples_split=2,
    #             min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
    #             oob_score=False, random_state=42, verbose=0, warm_start=False)


def build_and_train_rf(X_train, Y_train, use_grid_search=False):
    """
    Build a RandomForest classifier model. Parameters have been found with the help of GridSearch cross validation
    :param X_train: (pandas Dataframe) the X split of 'Training' dataset (i.e all features)
    :param Y_train: (pandas Dataframe) the Y split of 'Training' dataset (i.e the target)
    :return: sklearn RandomForest classifier as the ML model
    """
    if use_grid_search:
        grid_search_for_rf(train_df)
    model = RandomForestClassifier(n_estimators=400, max_depth=5.0, max_features=0.5)
    model.fit(X_train, Y_train)
    print("Score for Random Forest is {:.2f}%".format(model.score(X_train, Y_train) * 100))
    return model


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    train_df = pd.read_csv('./datasets/train.csv')
    test_df = pd.read_csv('./datasets/test.csv')

    # Build and keep some referentials for later use
    fare_means = build_fare_means(train_df, test_df)
    age_table = build_age_table(train_df, test_df)

    # Handle both training and test dataset in same way
    print("-----------------  Data cleaning  -----------------")
    train_df = handle_missing_values(train_df, age_table)
    test_df = handle_missing_values(test_df, age_table, fare_means)

    print("--------------  Feature engineering  --------------")
    train_df = transform_and_create_new_features(train_df)
    test_df = transform_and_create_new_features(test_df)

    print("---------  Build training and test splits  --------")
    train_split, test_split = train_test_split(train_df.drop(['PassengerId'], axis=1), test_size=0.25, random_state=42)
    X_train = train_split.drop(['Survived'], axis=1)
    Y_train = train_split['Survived']
    X_test = test_split.drop(['Survived'], axis=1)
    Y_test = test_split['Survived']

    # Try a Random Forest classification
    model = build_and_train_rf(X_train, Y_train)

    # Time for prediction !
    # Apply the trained model to the test data (omitting the column PassengerId) to produce an output of predictions.
    X_prediction = test_df.drop("PassengerId", axis=1).copy()
    output = model.predict(X_prediction)

    build_and_save_from_output(test_df, output, 'titanic_1-1-v4')
