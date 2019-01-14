"""
Created on 13/01/2019
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import pandas as pd


def build_fare_means(train_df, test_df):
    """
    In 'Test' dataset, there is 1 missing fare. Strategy is to replace it with the mean value according to the
    associated Pclass
    :param train_df: (pandas Dataframe) the 'Training' dataset
    :param test_df: (pandas Dataframe) the 'Test' dataset
    :return: (pandas Dataframe) table with mean fare per Pclass
    """
    return _build_concat_df(train_df, test_df).pivot_table('Fare', index='Pclass', aggfunc='mean')


def build_age_table(train_df, test_df):
    """
    As there seems to be a correlation among Age, Sex and Pclass, we build a table that will later be used to 'guess'
    the missing age for one people based on its Sex and Pclass
    :param train_df: (pandas Dataframe) the 'Training' dataset
    :param test_df: (pandas Dataframe) the 'Test' dataset
    :return: (dict) a table with key as format '#{SEX}#{PCLASS}' and value as the median value for age in the combination
    of both 'Training' and 'Test' dataset
    """
    overall_df = _build_concat_df(train_df, test_df)
    overall_df.Age.dropna()

    sexs = overall_df.Sex.unique()
    pclasses = overall_df.Pclass.unique()
    age_table = {}
    for sex in sexs:
        for pclass in pclasses:
            med_age = overall_df[(overall_df.Sex == sex) & (overall_df.Pclass == pclass)]['Age'].median()
            age_table['#{}#{}'.format(sex, pclass)] = med_age
    return age_table


def handle_missing_values(df, age_table, fare_means=None):
    """
    Drop useless data and fill missing values in the given dataset
    :param df: (pandas Dataframe) a given dataset on which we will apply our modifications
    :param age_table: (dict) a table with key as format '#{SEX}#{PCLASS}' and value as the median value for age in the
    combination of both 'Training' and 'Test' dataset
    :param fare_means: (pandas Dataframe) table with mean fare per Pclass. Default is None
    :return: (pandas Dataframe) a new dataset on which cleaning and other operations have been made
    """
    # As decided, drop those useless data
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # 'AGE' FEATURE MANAGEMENT
    # Near 100 'Age' values are missing in both datasets
    # One basic approach would be to replace null values by the mean value but problem is there were many people with
    # many different ages and so we cannot assign the mean age (~29 years old) to the 177 N/A !
    print("Before operation, mean is {}".format(df.Age.mean()))
    df['Age'] = df.apply(lambda x: age_table["#{}#{}".format(x['Sex'], x['Pclass'])] if pd.isnull(x['Age']) else x['Age'], axis=1)
    print("After operation, mean is {}".format(df.Age.mean()))

    # 'EMBARKED' FEATURE MANAGEMENT
    # In 'Training' dataset, some 'Embarked' informations are missing. There are few, we can replace with the most
    # common value found in the column (by using the 'mode' function)
    mode_embarked = df.Embarked.dropna().mode()[0]
    df['Embarked'] = df.Embarked.fillna(mode_embarked)

    # 'FARE' FEATURE MANAGEMENT
    # Only for 'Test' dataset: replace the missing fare value with the mean value according to the associated Pclass
    if fare_means is not None:
        df['Fare'] = df[['Fare', 'Pclass']].\
            apply(lambda x: fare_means.loc[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)

    return df


def _build_concat_df(train_df, test_df):
    """
    Build a new dataset which is the combination of both datasets given as parameters
    :param train_df: (pandas Dataframe) the 'Training' dataset
    :param test_df: (pandas Dataframe) the 'Test' dataset
    :return: (pandas Dataframe) a totally new dataset which is the combination of both 'Training' and 'Test' dataset
    """
    datasets = [train_df.drop(['Survived'], axis=1), test_df]
    return pd.concat(datasets)

