"""
Created on 22/12/2018
@author: nidragedd (based on pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

nb_rows = 5


def build_fare_means(train_df, test_df):
    """
    In test dataset, there is 1 missing fare. Strategy is to replace it with the mean value according to the associated
    Pclass
    :param train_df: (pandas Dataframe) the 'training' dataset
    :param test_df: (pandas Dataframe) the 'test' dataset
    :return: (pandas Dataframe) table with mean fare per Pclass
    """
    datasets = [train_df.drop(['Survived'], axis=1), test_df]
    overall_df = pd.concat(datasets)
    return overall_df.pivot_table('Fare', index='Pclass', aggfunc='mean')


def build_age_table(train_df, test_df):
    """
    As there is clear correlation among Age, Gender, and Pclass, we build a table that will later be used to 'guess' the
    missing age of a people based on its Sex and Pclass
    :param train_df: (pandas Dataframe) the 'training' dataset
    :param test_df: (pandas Dataframe) the 'test' dataset
    :return: (dict) a table with key as format '#{SEX}#{PCLASS}' and value as the median value for age in the combination
    of both 'Training' and 'Test' dataset
    """
    datasets = [train_df.drop(['Survived'], axis=1), test_df]
    overall_df = pd.concat(datasets)
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
    # One basis approach would replace by the mean value but problem is there were many people with many different ages
    # and so we cannot assign the mean age (~29 years old) to the 177 N/A !
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


def transform_and_create_new_features(df):
    """
    Performs transformations (map strings to numeric, new feature creation) on columns of the given dataset
    :param df: (pandas Dataframe) a given dataset on which we will apply our modifications
    :return: (pandas Dataframe) a new dataset on which operations have been made
    """
    # As we can only use numeric values, create a new column named 'Gender' and another one named 'Port', then drop
    # original features
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df['Port'] = df['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
    df = df.drop(['Sex', 'Embarked'], axis=1)

    return df


def build_and_save_from_output(test_df, output_array, filename):
    """
    Save predictions dataset on disk in a file with the given file name.
    :param test_df: (pandas Dataframe) the test dataset used to make predictions
    :param output_array: (ndarray) survived or not (as 1 or 0) predictions
    :param filename: (string) name of the CSV file to generate
    """
    # Memo: test_df.values[:, 0] = all rows but only first column
    # Memo: np.c_ will stack both arrays along their last axis --> in our case we have 2 arrays (418,) that will
    # produce a new one (418, 2)
    result = np.c_[test_df.values[:, 0].astype(int), output_array.astype(int)]

    # Build a pandas DataFrame object from this column-stacked array and perform a consistency check on its shape
    result_df = pd.DataFrame(result, columns=['PassengerId', 'Survived'])
    print("Total elements are (rows, columns): {}".format(result_df.shape))

    result_df.to_csv('./results/{}.csv'.format(filename), index=False)


if __name__ == '__main__':
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

    # Build the X_train and X_test which are respectively training and test dataset AND the Y_train which is the target
    # for training (which is column 'Survived' in our case)
    X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)  # Useless data for training
    Y_train = train_df["Survived"]  # Target in training dataset

    # Try a Random Forest classification
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)

    # Apply the trained model to the test data (omitting the column PassengerId) to produce an output of predictions.
    X_test = test_df.drop("PassengerId", axis=1).copy()
    output = random_forest.predict(X_test)
    rf_score = random_forest.score(X_train, Y_train) * 100
    print("Score for Random Forest is {:.2f}%".format(rf_score))

    build_and_save_from_output(test_df, output, 'titanic_1-1-v2')
