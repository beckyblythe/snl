from brian2 import *
from Neuron import *
from stat_testing import *

def parameters_by_name(name):

    if name == 'segments':
        keys = ['time_up', 'time_down', 'time_above']
        xlabel = 'time (ms)'
    if name == 'quiet_and_burst':
        keys = ['ISI_quiet', 'ISI_burst']
        xlabel = 'ISI (ms)'
    if name == 'burst':
        keys = ['ISI_burst']
        xlabel = 'ISI (ms)'
    if name == 'all':
        keys = ['ISI']
        xlabel = 'ISI (ms)'
    if name == 'quiet_and_up':
        keys = ['ISI_quiet', 'time_up']
        xlabel = 'time (ms)'
    if name == 'above_and_down':
        keys = ['time_above', 'time_down']
        xlabel = 'time (ms)'
    if name == 'burst_and_above':
        keys = ['ISI_burst', 'time_above']
        xlabel = 'time (ms)'
    if name == 'n_last_down':
        xlabel = 'n'
        keys = [name]
    if name == 'n_last_up':
        keys = [name]
        xlabel = 'n'
    if name == 'up_and_down':
        keys = ['time_up', 'time_down']
        xlabel = 'n'
    if name == 'time_above':
        keys = [name]
        xlabel = 'time (ms)'
    if name == 'up':
        keys = ['time_up']
        xlabel = 'time (ms)'

    
    ylabel = 'Distribution'    
    return keys, xlabel, ylabel

def plot_subplot(fig, file_name, name, keys, idx_tau_n, idx_Iapp, log, cut, fit):
    ax = fig.add_subplot(5,5,5*(4 - idx_tau_n)+idx_Iapp +1)
    
#    ax.set_ylim((10**(-2.8),10**0))
    try:            
        results = get_simulation(file_name)
        print(file_name)
        flat_results = np.array([result for neuron in results['ISI'] for result in neuron])
        means = {}
        var = {}
        for key in keys:

            flat_results = np.array([result for neuron in results[key] for result in neuron])*1000
            ax.hist(flat_results, normed = True, bins = 40,  alpha = .5, range = ((0,cut)), log = log, label = key)
            ax.set_xlim((0,cut))
#            plot_fit(flat_results, cut, fit)
            means[key] = int(flat_results.mean())
            means[key] = round(flat_results.mean(),2)
        means_title_part = ''.join(['\n' + key + ' mean: ' + str(means[key]) + 'ms' for key in keys])
        title = ''

        if name == 'quiet_and_burst':
            num_ISI = np.sum([len(neuron) for neuron in results['ISI']])
            num_ISI_burst = np.sum([len(neuron) for neuron in results['ISI_burst']])
            title = str(int(num_ISI_burst/num_ISI*100))+'% ISI_burst'
        if name == 'all':
            num_ISI = np.sum([len(neuron) for neuron in results['ISI']])

            ax.set_title(str(num_ISI) + ' ISIs' + means_title_part)
            
        if name == 'burst':
            ax.set_title(means_title_part)
        ax.set_ylim((0,1))
        ax.set_title(title+means_title_part)
    except IOError:
        ax.axis('off')
        pass
    

def plot_all_histograms(name, log = False, cut=20, fit = None):
    plt.rcParams['figure.figsize'] = 13, 16  
    tau_ns = [ .155, .1575,.16,.1625, .165]*ms
    Iapps = [1.2,2.3,3.2,3.9,4.3]*uA
    I_noises = [2,2.5,3]*uA
    duration = 50*second
    number = 10    
    keys, xlabel, ylabel = parameters_by_name(name)   
    if log:
        ylabel = 'log-' + ylabel
    
    for I_noise in I_noises:
        fig = plt.figure()
        for idx_tau_n, tau_n in enumerate(tau_ns):
            for idx_Iapp, Iapp in enumerate(Iapps):
                file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
                plot_subplot(fig, file_name, name, keys, idx_tau_n, idx_Iapp, log, cut, fit)
                
        plt.legend(loc = 1)
        plt.tight_layout()
        fig.text(.5, -.005, xlabel, ha='center', fontsize=12)
        fig.text(-.005, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)
#        
        
        plt.savefig('pictures_report/'+name+' '+str(I_noise)+'.png', bbox_inches = 'tight')
        
plot_all_histograms('all', log = False, cut = 20)

