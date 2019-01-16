"""
Created on 16/01/2019
@author: nidragedd

Basic Neural Network implementation for this specific problem
"""
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from data.data_transformation import specific_normalization
from utils import display


def build_and_train_nn(train_df):
    """
    Build a basic Keras Neural Network model. Parameters are kind of arbitrary (could be better to use GridSearch to
    find them).
    :param train_df: (pandas Dataframe) the whole 'Training' dataset
    :return: a keras object as the ML model
    """
    train_df = specific_normalization(train_df)
    X_train = train_df.drop(['Survived'], axis=1)
    Y_train = train_df['Survived']

    model = Sequential()
    model.add(Dense(30, input_shape=(X_train.shape[1], ), activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
    # To avoid over-fitting
    model.add(Dropout(0.2))
    # Sigmoid for output layer as it suits binary classification => outputs will be between 0 and 1
    model.add(Dense(1, activation="sigmoid"))

    # Hyperparameters settings
    learning_rate = 0.01
    nb_epochs = 50
    optimizer = SGD(lr=learning_rate)  # SGD (Stochastic Gradient Descent)
    loss_function = "binary_crossentropy"

    print("Compiling Neural Network model and train it with 80/20 CV split")
    model.compile(loss=loss_function, optimizer='adam', metrics=["accuracy"])
    fit_history = model.fit(X_train, Y_train, epochs=nb_epochs, validation_split=0.2)

    display.display_plot_history(nb_epochs, fit_history)

    return model
