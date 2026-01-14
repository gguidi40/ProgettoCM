import matplotlib.pyplot as plt

class dual_plot:
    def __init__(self):
        pass

    def plot_error(self, err, val):
        plt.figure('Training, validation loss')
        plt.plot(err, c='red', linestyle='-', label='Average training loss')
        plt.plot(val, c='blue', linestyle='-', label='Average validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()
    
    def plot_metric(self, err, val):
        plt.figure('KLD, SSIM')
        plt.plot(err, c='red', linestyle='-', label='KLD')
        plt.plot(val, c='blue', linestyle='-', label='SSIM')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.grid()
        plt.legend()
        plt.show()