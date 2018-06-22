from Neuron import get_simulation
from brian2 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = 12, 9

tau_ns = [.165,.1625,.160,.1575,.155]*ms
Iapps = [4.3,3.9,3.2,2.3,1.2]*uA
I_noises = [2,2.5,3]*uA
duration = 50000*ms
number = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('tau_n (ms)')
ax.set_ylabel('Iapp (uA)')
ax.set_zlabel('fraction of burtst ISIs')

ax.set_xticks([.165,.1625,.160,.1575,.155])
ax.set_yticks([4.3,3.9,3.2,2.3,1.2])
ax.view_init(10,-75)
plt.gca().invert_xaxis()

file = open('ratio.txt', 'w+')
for I_noise, color in zip(I_noises,['b','r','g']):
    for tau_n in tau_ns:
        for Iapp in Iapps:
            file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
           
            
            try:     
                data = get_simulation(file_name)
                print(file_name)
                ISIs = [ISI for neuron in data['ISI'] for ISI in neuron ]
                ISI_quiet = [ISI for neuron in data['ISI_burst'] for ISI in neuron ]
                ratio = len(ISI_quiet)/len(ISIs)
                print(tau_n/ms,Iapp/uA, ratio)    
                file.write(str(tau_n/ms)+','+str(Iapp/uA) +','+str(round(ratio,2))+';')
                    
                    
                
            except IOError:
                print('1')
                pass
file.close()
