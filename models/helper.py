"""
Created on 14/01/2019
@author: nidragedd

Helper file to factorize code about GridSearchCV usage or dataset splitting
"""
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split


def grid_search_tuning(params, classifier, X_train, Y_train):
    """
    Build a GridSearchCV for a given classifier and display best tuning parameters to use them later
    :param params: (dict) the parameters to use for searching
    :param classifier: (sklearn object) the classifier for which we will try to tune the parameters on the given dataset
    :param X_train: (pandas Dataframe) the X split of 'Training' dataset (i.e all features)
    :param Y_train: (pandas Dataframe) the Y split of 'Training' dataset (i.e the target)
    :return: best classifier found
    """
    gs = GridSearchCV(estimator=classifier, param_grid=params, verbose=True, cv=5)
    gs.fit(X_train, Y_train)
    print("Grid search best score is {}".format(gs.best_score_))
    print("Grid search best estimator is {}".format(gs.best_estimator_))
    return gs.best_estimator_


def split_data(train_df):
    """
    Helper method to wrap the train_test_split function from sklearn package. Will return:
     * (X, Y) as respectively features and targets for the whole dataset
     * (X_train, Y_train): same but on the 'train' split part
     * (X_test, Y_test): same but on the 'test' split part
    :param train_df: the whole 'Training' dataset
    :return: 6-tuple object
    """
    train_split, test_split = train_test_split(train_df, test_size=0.3, random_state=42)
    X_train = train_split.drop(['Survived'], axis=1)
    Y_train = train_split['Survived']
    X_test = test_split.drop(['Survived'], axis=1)
    Y_test = test_split['Survived']

    X = train_df.drop(['Survived'], axis=1)
    Y = train_df['Survived']

    return (X, Y, X_train, Y_train, X_test, Y_test)


def fit_and_score(model_name, model, X_train, Y_train, X_test, Y_test):
    """
    Helper method used to wrap the 'fit' method + 'predict' on the test split part in order to display some accuracy
    metric
    :param model_name: (string) the model name (as given as argument parameter of the program)
    :param model: (object) the model itself
    :param X_train: (pandas Dataframe) features of the train dataset
    :param Y_train: (pandas Dataframe) target of the train dataset
    :param X_test: (pandas Dataframe) features of the test split part of the dataset
    :param Y_test: (pandas Dataframe) target of the test split part of the dataset
    """
    # Fit on the whole dataset (i.e not the "splitted" part)
    model.fit(X_train, Y_train)
    # Performance on a known part of the dataset (i.e the 'test split' part)
    prediction = model.predict(X_test)

    print("Overall score for {} is {:.2f}%".format(model_name, model.score(X_train, Y_train) * 100))
    print('Based on test split, score is {:.2f}%'.format(metrics.accuracy_score(prediction, Y_test) * 100))
