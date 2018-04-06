from brian2 import *
from Neuron import *

def parameters_by_name(name):
    if name == 'segments':
        keys = ['time_up', 'time_down', 'time_above']
    if name == 'quiet_and_burst':
        keys = ['ISI_quiet', 'ISI_burst']
    if name == 'burst':
        keys = ['ISI_burst']
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
        
    return keys

def plot_subplot(fig, file_name, name, idx_tau_n, idx_Iapp):
    keys = parameters_by_name(name)
    
    ax = fig.add_subplot(5,5,5*(4 - idx_tau_n)+idx_Iapp +1)
    
    try:     
       
        results = get_simulation(file_name)
        print(file_name)
        intervals = results
        flat_intervals = np.array([interval for neuron in intervals['ISI'] for interval in neuron])
        mean = 5 # flat_intervals.mean()*1000
        ax.xlim = ((0,5*mean))
        for key in keys:
            flat_intervals = np.array([interval for neuron in intervals[key] for interval in neuron])
            ax.hist(flat_intervals*1000, normed = True, bins = 50,  alpha = .5, range = ((0,5*mean)), log = False, label = key)
        if name == 'quiet_and_burst':
            num_ISI = np.sum([len(neuron) for neuron in intervals['ISI']])
            num_ISI_burst = np.sum([len(neuron) for neuron in intervals['ISI_burst']])
            ax.set_title(str(int(num_ISI_burst/num_ISI*100))+'% ISI_burst')
    except IOError:
        ax.axis('off')
        pass
    

def plot_all_histograms(name):
    plt.rcParams['figure.figsize'] = 12, 12  

    tau_ns = [ .155, .1575,.16,.1625, .165]*ms
    Iapps = [1.2,2.3,3.2,3.9,4.3]*uA
    I_noises = [2,2.5,3]*uA
    duration = 50*second
    number = 10
          
    for I_noise in I_noises:
        
        fig = plt.figure()
        
        for idx_tau_n, tau_n in enumerate(tau_ns):
            for idx_Iapp, Iapp in enumerate(Iapps):
                file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
                plot_subplot(fig, file_name, name, idx_tau_n, idx_Iapp )
                
                
        plt.legend(loc = 1)
       
        fig.text(0.5, -0.01, 'common X', ha='center')
        fig.text(-0.01, 0.5, 'common Y', va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig('pictures_report/'+name+' '+str(I_noise)+'.png', bbox_inches='tight')
        
plot_all_histograms('quiet_and_burst')