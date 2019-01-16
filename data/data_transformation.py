"""
Created on 13/01/2019
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform_and_create_new_features(df):
    """
    Performs transformations (map strings to numeric, new feature creation) on columns of the given dataset
    :param df: (pandas Dataframe) a given dataset on which we will apply our modifications
    :return: (pandas Dataframe) a new dataset on which operations have been made
    """
    # 'GENDER' FEATURE MANAGEMENT
    # Transform 'Gender' feature (categorical) to numerical one
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 'EMBARKED' FEATURE MANAGEMENT
    # 1st approach: df['Port'] = df['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
    # Extract from 'pycon UK Tutorial':
    # "Replacing {C, S, Q} by {1, 2, 3} would seem to imply the ordering C < S < Q when in fact they are simply arranged
    # alphabetically. To avoid this problem, we create dummy variables. Essentially this involves creating new columns
    # to represent whether the passenger embarked at C with the value 1 if true, 0 otherwise."
    dummies_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, dummies_embarked], axis=1)

    # 'AGE' & 'FARE' FEATURES MANAGEMENT
    df = _transform_age_feature(df)
    df = _transform_fare_feature(df)

    # CREATION OF A NEW FEATURE: Family size + Alone or not ?
    df['Family'] = df['SibSp'] + df['Parch']
    df['Alone'] = 0
    df.loc[df['Family'] == 0, 'Alone'] = 1

    # Drop all columns that are now useless
    df = df.drop(['Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch'], axis=1)
    print(df.head(10))

    return df


def _transform_age_feature(df):
    """
    Transform the 'Age' feature by splitting the Age values into arbitrary ranges and then creating dummies so that our
    model will not put more importance on 8 value than 5 for example.
    :param df: (pandas Dataframe) a given dataset on which we will apply our modifications
    :return: (pandas Dataframe) a new dataset on which operations have been made
    """
    df = df.apply(_build_age_range, axis='columns')
    dummies_age = pd.get_dummies(df['Age'], prefix='Age')
    print("For dataset with shape {}, the dummies for 'Age' are: {}".format(df.shape, dummies_age.columns))
    df = pd.concat([df, dummies_age], axis=1)

    # Ensure that all dummies are created and that 'Training' and 'Test' datasets will have same number of columns. In
    # our case, 'Age_8' will not be created for 'Test' dataset. We could create it by hand but it is more robust to test
    # all cases
    # For 'Age', range has been splitted in 8
    for i in range(8):
        if 'Age_{}'.format(i) not in df:
            df['Age_{}'.format(i)] = 0

    return df


def _transform_fare_feature(df):
    """
    Transform the 'Fare' feature which is a 'continuous' variable into a categorical one (with dummies so that there is
    no importance for the order)
    :param df: (pandas Dataframe) a given dataset on which we will apply our modifications
    :return: (pandas Dataframe) a new dataset on which operations have been made
    """
    df = df.apply(_build_fare_range, axis='columns')
    dummies_fare = pd.get_dummies(df['Fare'], prefix='Fare')
    print("For dataset with shape {}, the dummies for 'Fare' are: {}".format(df.shape, dummies_fare.columns))
    df = pd.concat([df, dummies_fare], axis=1)

    # 'Fare' has been splitted in 4
    for i in range(4):
        if 'Fare_{}'.format(i) not in df:
            df['Fare_{}'.format(i)] = 0

    return df


def _build_age_range(row):
    """
    This is an arbitrary cut of the whole Age feature range
    :param row: the row to update
    :return: updated row with a string value in AgeRange feature based on its value in Age
    """
    val = 0
    if row.Age <= 10:
        val = 0
    elif 10 < row.Age <= 15:
        val = 1
    elif 15 < row.Age <= 25:
        val = 2
    elif 25 < row.Age <= 40:
        val = 3
    elif 40 < row.Age <= 60:
        val = 4
    elif 60 < row.Age <= 70:
        val = 5
    elif 70 < row.Age <= 77:
        val = 6
    elif row.Age > 77:
        val = 7
    elif pd.isnull(row.Age):
        val = 9

    row.Age = val
    return row


def _build_fare_range(row):
    """
    When we cut the Fare feature in 4, we got those ranges: (-0.001, 7.91], (7.91, 14.454], (14.454, 31.0], (31.0, 512.329]
    :param row: the row to update
    :return: updated row with a new value in Fare feature based on its value before entering this function
    """
    val = 0
    if 7.91 < row.Fare <= 14.454:
        val = 1
    elif 14.454 < row.Fare <= 31:
        val = 2
    elif row.Fare > 31:
        val = 3
    row.Fare = val
    return row


def specific_normalization(df):
    """
    Depending on the model, sometimes we cannot afford having values which are not normalized. A standard approach is
    to scale the inputs to have mean 0 and a variance of 1
    :param df: (pandas Dataframe) the dataset to scale
    :return: (pandas Dataframe) the scaled dataset
    """
    # Need to scale some vars. This is done using a StandardScaler from sklearn package
    scaler = StandardScaler()
    df['Pclass'] = df['Pclass'].astype('float64')
    df['Family'] = df['Family'].astype('float64')
    # .reshape(-1, 1) is mandatory otherwise an exception is thrown (as 'data has a single feature')
    df['Pclass'] = scaler.fit_transform(df['Pclass'].values.reshape(-1, 1))
    df['Family'] = scaler.fit_transform(df['Family'].values.reshape(-1, 1))

    return df
