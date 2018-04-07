from brian2 import *
from Neuron import *

def plot_all_histograms(name, log=False):
    plt.rcParams['figure.figsize'] = 12, 12  

    tau_ns = [ .155, .1575,.16,.1625, .165]*ms
    Iapps = [1.2,2.3,3.2,3.9,4.3]*uA
    I_noises = [2,2.5,3]*uA
    duration = 50*second
    number = 10
    
    if name == 'segments':
        keys = ['time_up', 'time_down', 'time_above']
    if name == 'quiet_and_burst':
        keys = ['ISI_quiet', 'ISI_burst']
    if name == 'all':
        keys = ['ISI']
    if name == 'quiet_and_up':
        keys = ['ISI_quiet', 'time_up']
    if name == 'above_and_down':
        keys = ['time_above', 'time_down']
    if name == 'burst_and_above':
        keys = ['ISI_burst', 'time_above']
    if name == 'n_last_down':
        keys = [name]
    if name == 'n_last_up':
        keys = [name]
        
    for I_noise in I_noises:
        
        fig = plt.figure()
        
        for idx_tau_n, tau_n in enumerate(tau_ns):
            for idx_Iapp, Iapp in enumerate(Iapps):
                file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
                ax = fig.add_subplot(5,5,5*(4 - idx_tau_n)+idx_Iapp +1)
                print(file_name)
                try:     
                    results = get_simulation(file_name)
                    intervals = results
                    flat_intervals = np.array([interval for neuron in intervals['ISI'] for interval in neuron])
                    mean = flat_intervals.mean()*1000
                    ax.xlim = ((0,5*mean))
                    for key in keys:
                        flat_intervals = np.array([interval for neuron in intervals[key] for interval in neuron])
                        ax.hist(flat_intervals*1000, normed = True, bins = 50,  alpha = .5, range = ((0,5*mean)), log = log, label = key)
                        
                except IOError:
                    ax.axis('off')
                    pass
                
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('pictures_report/'+name+' '+str(I_noise)+'.png')
        
plot_all_histograms('quiet_and_burst', log = False)