import numpy as np
import matplotlib.pyplot as plt   
from scipy import *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

def inverse_gauss(x,Lambda, mu):
    return np.sqrt(Lambda/(2*pi*x**3))*np.exp(-Lambda*(x-mu)**2/(2*mu**2*x))

def exponential(x,Lambda):
    return Lambda * exp(-Lambda*x)


def plot_fit(flat_results, cut, fit):
    mu = flat_results.mean()
    x=np.arange(.1,cut,.05)
    if fit == 'exp':
        Lambda = 1/ mu
        plt.plot(x, exponential(x, Lambda))
    if fit == 'inv':
        Lambda = 1 / (np.mean(1/flat_results) - 1 / mu)
        plt.plot(x, inverse_gauss(x,Lambda,mu))
    return