from Neuron import get_simulation
from brian2 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = 9, 6

tau_n = .153*ms
Iapp = 1* uA #/cm**2
I_noise = 2.5*uA
duration = 50000*ms

v0=-30*mV
n0=.05

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('tau_n (ms)')
ax.set_ylabel('Iapp (uA)')
ax.set_zlabel('fraction of burtst ISIs')
ax.set_xticks([.155, .1575,.16,.1625, .165])
ax.set_yticks([1.2,2.3,3.2,3.9,4.3,4.51])
ax.view_init(10,180)

for tau_n in [ .155, .1575,.16,.1625, .165]*ms:
    for Iapp in [1.2,2.3,3.2,3.9,4.3,4.51]*uA:
        for duration in [10000,50000]*ms:

            file_name=str(tau_n)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(duration/second)+' s  '+str(I_noise)
#            print(file_name)
            
            try:     
                data = get_simulation(file_name)
                ISIs = [ISI for neuron in data['ISI'] for ISI in neuron ]
                ISI_quiet = [ISI for neuron in data['ISI_quiet'] for ISI in neuron ]
                ratio = len(ISI_quiet)/len(ISIs)
                print(tau_n/ms,Iapp/uA, ratio)    
                ax.scatter(tau_n/ms,Iapp/uA, ratio)
                
            
            except IOError:
                pass
