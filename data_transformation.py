"""
Created on 13/01/2019
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import pandas as pd


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

    # 'AGE' FEATURE MANAGEMENT
    df['AgeRange'] = df['Age']
    df = df.apply(_build_age_range, axis='columns')
    df['Age'] = df['AgeRange'].\
        map({'0-10': 1, '10-15': 2, '15-25': 3, '25-40': 4, '40-60': 5, '60-70': 6, '70-77': 7, '>77': 8}).astype(int)

    # 'FARE' FEATURE MANAGEMENT
    df = df.apply(_build_fare_range, axis='columns')
    df['Fare'] = df['Fare'].astype(int)

    # CREATION OF A NEW FEATURE: Family size + Alone or not ?
    df['Family'] = df['SibSp'] + df['Parch']
    df['Alone'] = 0
    df.loc[df['Family'] == 0, 'Alone'] = 1

    # Drop all columns that are now useless
    df = df.drop(['Sex', 'Embarked', 'AgeRange', 'SibSp', 'Parch'], axis=1)
    print(df.head(10))

    return df


def _build_age_range(row):
    """
    This is an arbitrary cut of the whole Age feature range
    :param row: the row to update
    :return: updated row with a string value in AgeRange feature based on its value in Age
    """
    val = ''
    if 0 < row.Age <= 10:
        val = '0-10'
    elif 10 < row.Age <= 15:
        val = '10-15'
    elif 15 < row.Age <= 25:
        val = '15-25'
    elif 25 < row.Age <= 40:
        val = '25-40'
    elif 40 < row.Age <= 60:
        val = '40-60'
    elif 60 < row.Age <= 70:
        val = '60-70'
    elif 70 < row.Age <= 77:
        val = '70-77'
    elif pd.isnull(row.Age):
        val = 'N/A'
    else:
        val = '>77'
    row.AgeRange = val
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
