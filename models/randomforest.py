"""
Created on 14/01/2019
@author: nidragedd

RandomForest implementation for this specific problem
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_for_rf(X_train, Y_train):
    """
    Build a GridSearchCV for RandomForest classifier and display best tuning parameters to use them later
    :param X_train: (pandas Dataframe) the X split of 'Training' dataset (i.e all features)
    :param Y_train: (pandas Dataframe) the Y split of 'Training' dataset (i.e the target)
    :return: sklearn RandomForest best classifier found
    """
    # Hyper-parameters management
    hyperparams = {
        'n_estimators': range(100, 1000, 100),
        'max_features': [0.5, 1.],
        'max_depth': [5., None]
    }

    gs = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=hyperparams, verbose=True, cv=5)
    gs.fit(X_train, Y_train)
    print("Grid search best score is {}".format(gs.best_score_))
    print("Grid search best estimator is {}".format(gs.best_estimator_))
    return gs.best_estimator_


def build_and_train_rf(train_df, use_grid_search=False):
    """
    Build a RandomForest classifier model. Parameters have been found with the help of GridSearchCV.
    Otherwise, setting the last argument to True will build a new classifier using Hyper-Parameter Tuning
    :param train_df: (pandas Dataframe) the whole 'Training' dataset
    :return: sklearn RandomForest classifier as the ML model
    """
    model = RandomForestClassifier(n_estimators=900, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                   max_features=0.5)
    X_train = train_df.drop(['Survived'], axis=1)
    Y_train = train_df['Survived']
    if use_grid_search:
        model = grid_search_for_rf(X_train, Y_train)
    model.fit(X_train, Y_train)
    print("Score for Random Forest is {:.2f}%".format(model.score(X_train, Y_train) * 100))
    return model
