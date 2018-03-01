from Neuron import get_points
import numpy as np

tau_n = .155*ms
Iapp = 1.2* uA #/cm**2
I_noise = 2*uA
duration = 10000*ms

v0=-30*mV
n0=.05

_, saddle, sep_slope, _ = get_points(tau_n, Iapp)

file_name=str(tau_n)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(duration/second) +' s  '+str(I_noise)

with open('simulations/'+file_name, 'rb') as f:
    results = pickle.load(f)
    
n_last_down = np.array([n for neuron in results['n_last_down'] for n in neuron])
n_last_up = np.array([n for neuron in results['n_last_up'] for n in neuron])


V_last_down = (n_last_down - saddle[1])*sep_slope[0]/sep_slope[1] + saddle[0]


plt.figure(figsize=(3., 3.))
plt.hist(n_last_down, bins = 15)
plt.xlabel('n')
plt.ylabel ('Distribution of n')
plt.savefig('down_distr/'+file_name+'.png', bbox_inches='tight') 