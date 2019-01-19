"""
Created on 22/12/2018
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import argparse
import datetime

import pandas as pd

from data.data_preparation import build_fare_means, build_age_table, handle_missing_values
from data.data_transformation import transform_and_create_new_features, specific_normalization
from models.deep import build_and_train_nn
from models.randomforest import build_and_train_rf
from models.xgb import build_and_train_xgb
from utils import constants
from utils.load_and_save import build_and_save_from_output


def prepare_data():
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

    print("----------  Saving transformed datasets  ----------")
    train_df.to_csv('./datasets/train-transformed.csv', index=False)
    test_df.to_csv('./datasets/test-transformed.csv', index=False)


def train_and_predict(model_name):
    train_df = pd.read_csv('./datasets/train-transformed.csv')
    test_df = pd.read_csv('./datasets/test-transformed.csv')

    train_df = train_df.drop(['PassengerId'], axis=1)

    model = None
    if model_name == constants.MODEL_RANDOMFOREST:
        model = build_and_train_rf(train_df)
    elif model_name == constants.MODEL_NEURALNETWORK:
        model = build_and_train_nn(train_df)
    elif model_name == constants.MODEL_XGBOOST:
        model = build_and_train_xgb(train_df)

    # Specific normalization for models that requires it
    if model_name in constants.NORMALIZATION_MODEL:
        test_df = specific_normalization(test_df)

    # Time for prediction !
    # Apply the trained model to the test data (omitting the column PassengerId) to produce an output of predictions.
    X_prediction = test_df.drop("PassengerId", axis=1).copy()
    predictions = model.predict(X_prediction)

    ts = f"{datetime.datetime.now():%Y-%m-%d-%H%M%S}"
    build_and_save_from_output(test_df, predictions, 'titanic_{}-{}'.format(model_name, ts))


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Mandatory arguments to run the program
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Name of the model, must be a valid choice")
    ap.add_argument("-o", "--objective", required=True, help="Choice between [prepare, predict, both]")
    args = vars(ap.parse_args())

    if args["model"] not in constants.ALLOWED_MODELS:
        raise Exception("Model must be a choice between those values: {}".format(constants.ALLOWED_MODELS))
    if args["objective"] not in constants.OBJECTIVE_OPTIONS:
        raise Exception("Objective must be a choice between those values: {}".format(constants.OBJECTIVE_OPTIONS))

    if args["objective"] == 'prepare' or args["objective"] == 'both':
        prepare_data()
    else:
        train_and_predict(args["model"])
