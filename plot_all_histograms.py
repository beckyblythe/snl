from brian2 import *
from Neuron import *

tau_ns = [ .155, .1575,.16,.1625, .165]*ms
Iapps = [1.2,2.3,3.2,3.9,4.3]*uA
I_noises = [2,2.5,3]*uA
duration = 50*second
number = 40

for I_noise in I_noises:
    for tau_n in tau_ns:
        for Iapp in Iapps:
            file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
            print(file_name)
            try:     
                results = get_simulation(file_name)
                plot_histograms(results) 
                if duration/ms >= 10000:
                    plt.savefig('histograms/'+file_name+'.png')
                plt.show()
            except IOError:
                pass