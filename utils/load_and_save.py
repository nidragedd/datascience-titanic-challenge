"""
Created on 13/01/2019
@author: nidragedd

This work is highly inspired by few good readings:
 * pycon UK Tutorial found here: https://github.com/savarin/pyconuk-introtutorial
 * EDA DieTanic found here: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 * Kaggle kernel from Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""


def build_and_save_from_output(test_df, predictions, filename):
    """
    Save predictions dataset on disk in a file with the given file name.
    :param test_df: (pandas Dataframe) the test dataset used to make predictions
    :param predictions: (ndarray) survived or not (as 1 or 0 OR values between 0 and 1) predictions
    :param filename: (string) name of the CSV file to generate
    """
    test_df['Survived'] = predictions
    test_df['Survived'] = test_df['Survived'].apply(lambda x: round(x, 0)).astype('int')
    result_df = test_df[['PassengerId', 'Survived']]
    print("Total elements are (rows, columns): {}".format(result_df.shape))
    result_df.to_csv('./results/{}.csv'.format(filename), index=False)
