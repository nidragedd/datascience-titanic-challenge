"""
Created on 16/01/2019
@author: nidragedd

Utils package to perform some useful plotting/displaying operations
"""
import matplotlib.pyplot as plt
import numpy as np


def display_plot_history(display_range, fit_history):
    """
    Build and display a graph corresponding to the given fit_history
    :param display_range: (int) number of epochs the model has been trained on
    :param fit_history: (keras object) the fitting history from where we will extract datas
    """
    np_range = np.arange(0, display_range)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np_range, fit_history.history["loss"], label="train_loss")
    plt.plot(np_range, fit_history.history["val_loss"], label="val_loss")
    plt.plot(np_range, fit_history.history["acc"], label="train_acc")
    plt.plot(np_range, fit_history.history["val_acc"], label="val_acc")
    plt.title("Training 'loss' and 'accuracy' metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy value")
    plt.legend()
    plt.show()
