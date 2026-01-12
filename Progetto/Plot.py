import numpy as np
import matplotlib.pyplot as plt

class error_plot:

    def plot(self, epochs, err):
        raise NotImplementedError

class dual_plot(error_plot):

    def plot(self, val, err):
        plt.figure('Validation, Error Plot')
        plt.plot(err, c='red', linestyle='-', label='Training')
        plt.plot(val, c='blue', linestyle='-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()