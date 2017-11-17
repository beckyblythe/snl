from scipy import *
from brian2 import *
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def inverse_gauss(x):
    return sqrt(Lambda/(2*pi*x**3))*exp(-Lambda*(x-mu)**2/(2*mu**2*x))


    

tau_n = .1575*ms
Iapp = 3.2* uA #/cm**2
I_noise = 2.5*uA
duration = 10000*ms

v0=-30*mV
n0=.05

file_name=str(tau_n)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(duration/second) +' s  '+str(I_noise)

with open('simulations/'+file_name, 'rb') as f:
    results = pickle.load(f)
    
ISI = np.array([interval for neuron in results['ISI'] for interval in neuron])*1000

mu = np.mean(ISI)
Lambda = np.mean(1/(1/ISI - 1/mu))
#
#print(stats.kstest(ISI, inverse_gauss))
#
#plt.figure()
#plt.hist(ISI, bins = 1000, normed = True)
#x=np.arange(0,20,.05)
#plt.plot(x, inverse_gauss(x))
#plt.axvline(mu)
#plt.xlim((0,20))
#plt.show()