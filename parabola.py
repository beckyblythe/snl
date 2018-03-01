from Neuron import simulate_neuron
import numpy as np

plt.rcParams['figure.figsize'] = 9, 9  

v0=[-70,-30]*mV
n0=[.2,.2]
I_noise = 0*uA
number = 2

duration = 20*ms

resolution = 30

tau_ns = np.linspace(.170,.1525, resolution)*ms
Iapps = np.linspace(.5,4.5,resolution)* uA #/cm**2

I_bif = np.empty(resolution)

for i,tau_n in enumerate(tau_ns):
    max_jump=0
    last_diff=0
    for Iapp in Iapps:
        _,_,_,V,n = simulate_neuron(tau_n, Iapp, number, v0, n0, duration, I_noise)
        print(min(V[1]),max(V[0]))
        if min(V[1])-max(V[0])-last_diff > max_jump:
            max_jump = min(V[1])-max(V[0])
            I_bif[i] = Iapp
        last_diff = min(V[1])-max(V[0])   