import matplotlib.pyplot as plt
import numpy as np


def plotNN_losses(array_list, labels = ['complex', 'simple'], colors = ['blue', 'orange']):
    
    """ Function plots N ( n_epochs x n_mods ) arrays on same graph with error bars
    Args:
        array_list:list of arrays of same dimensions, (no. epochs x no. models)
        labels: labels for each array
        colors:
        errspace: number epochs seperating the errorbars
    Returns:
        pyplot of N lines with error bars
    """
    for i, arr in enumerate(array_list):
        x = np.arange(0,arr.shape[0],1)
        mn = arr.mean(axis=1)
        errs = arr.std(axis=1)#[::errspace]
        
        plt.plot(x, mn, label = labels[i], color = colors[i])
        plt.fill_between(x, mn - errs, mn + errs, alpha = 0.3, facecolor = colors[i])
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()


def plotNN_shifted(array_list, labels = ['complex', 'simple'], colors = ['blue', 'orange'], shift = 300):
    
    """ Function plots N ( n_epochs x n_mods ) arrays on same graph with error bars
    Args:
        array_list:list of arrays of same dimensions, (no. epochs x no. models)
        labels: labels for each array
        colors:
        errspace: number epochs seperating the errorbars
    Returns:
        pyplot of N lines with error bars
    """
    for i, arr in enumerate(array_list):
        x = np.arange(0,arr.shape[0],1) + shift
        mn = arr.mean(axis=1)
        errs = arr.std(axis=1)
        
        plt.plot(x, mn, label = labels[i], color = colors[i])
        plt.fill_between(x, mn - errs, mn + errs, alpha = 0.3, facecolor = colors[i])
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()


