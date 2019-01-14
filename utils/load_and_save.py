"""
Created on 13/01/2019
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""
import pandas as pd
import numpy as np


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
