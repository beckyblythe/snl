import numpy as np
from brian2 import *
from Neuron import *

tau_ns = [ .155, .1575,.16,.1625, .165]*ms
Iapps = [1.2,2.3,3.2,3.9,4.3]*uA
I_noises = [2,2.5,3]*uA
duration = 50*second
number = 10

for I_noise in I_noises:
    for tau_n in tau_ns:
        for Iapp in Iapps:
            file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
            print(file_name)
            try:     

                with open('simulations/'+file_name, 'rb') as f:
                    results = get_simulation(file_name)
                    _, saddle,_,_ = get_points(tau_n, Iapp)
    
                    n_last_down = np.array([n for neuron in results['n_last_down'] for n in neuron])
                    n_last_up = np.array([n for neuron in results['n_last_up'] for n in neuron])


                    plt.figure(figsize=(3., 3.))
                    plt.hist(n_last_down, bins = 30)
                    plt.axvline(saddle[1], color = '.5')
                    plt.axvline(np.median(n_last_down), color = 'b')
                    plt.xlabel('n')
                    plt.ylabel ('Distribution of n')
                    plt.xlim((-.1,.7))
                    plt.savefig('down_distr/'+file_name+'.png', bbox_inches='tight') 
                    plt.show()
                    plt.close()
                    
                    plt.figure(figsize=(3., 3.))
                    plt.hist(n_last_up, bins = 30, range = ((0,.003)))
                    plt.axvline(saddle[1], color = '.5')
                    plt.axvline(n_last_up.mean(), color = 'r')
                    plt.xlabel('n')
                    plt.ylabel ('Distribution of n')
                    plt.xlim((0,.003))
                    plt.savefig('up_distr/'+file_name+'.png', bbox_inches='tight') 
                    plt.show()
                    plt.close()
                    
            except IOError:
                pass