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
        xlabel = 't (ms)'
    if name == 'up':
        keys = ['time_up']
        xlabel = 't (ms)'

    
    ylabel = 'Distribution'    
    return keys, xlabel, ylabel

def plot_subplot(fig, file_name, name, keys, idx_tau_n, idx_Iapp, log, cut, fit):
    ax = fig.add_subplot(5,5,5*(4 - idx_tau_n)+idx_Iapp +1)
    ax.set_xlim((0,cut))
#    ax.set_ylim((10**(-2.8),10**0))
    try:            
        results = get_simulation(file_name)
        print(file_name)
        flat_results = np.array([result for neuron in results['ISI'] for result in neuron])
        means = {}
        var = {}
        for key in keys:
<<<<<<< HEAD
            flat_results = np.array([result for neuron in results[key] for result in neuron])
            ax.hist(flat_results*1000, normed = True, bins = 50,  alpha = .5, range = ((0,cut)), log = False, label = key)
            means[key] = round(flat_results.mean()*1000,2)
            var[key] = round(np.var(flat_results*1000),2)
        means_title_part = ''.join(['\n' + key + ' mean: ' + str(means[key]) + 'ms\n'+'var: ' + str(var[key]) + 'ms^2'for key in keys])
=======
            flat_results = np.array([result for neuron in results[key] for result in neuron])*1000
            ax.hist(flat_results, normed = True, bins = 50,  alpha = .5, range = ((0,cut)), log = log, label = key)
            
            plot_fit(flat_results, cut, fit)
                
            means[key] = int(flat_results.mean())
            means[key] = round(flat_results.mean(),2)
        means_title_part = ''.join(['\n' + key + ' mean: ' + str(means[key]) + 'ms' for key in keys])
        title = ''
>>>>>>> 2aaf0206243d36e03c9603dc81dcafdc1f6422e6
        if name == 'quiet_and_burst':
            num_ISI = np.sum([len(neuron) for neuron in results['ISI']])
            num_ISI_burst = np.sum([len(neuron) for neuron in results['ISI_burst']])
            title = str(int(num_ISI_burst/num_ISI*100))+'% ISI_burst'
        if name == 'all':
            num_ISI = np.sum([len(neuron) for neuron in results['ISI']])
            
<<<<<<< HEAD
            ax.set_title(str(num_ISI) + ' ISIs' + means_title_part)
            
        if name == 'burst':
            ax.set_title(means_title_part)
=======
            title = str(num_ISI) + ' ISIs' 
            
        ax.set_title(title + means_title_part)
>>>>>>> 2aaf0206243d36e03c9603dc81dcafdc1f6422e6
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
    
    for I_noise in I_noises:
        fig = plt.figure()
        for idx_tau_n, tau_n in enumerate(tau_ns):
            for idx_Iapp, Iapp in enumerate(Iapps):
                file_name=str(duration/second)+' s  '+str(I_noise)+' '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
                plot_subplot(fig, file_name, name, keys, idx_tau_n, idx_Iapp, log, cut, fit)
                
        plt.legend(loc = 1)
        fig.text(0.5, -.02, xlabel, ha='center')
        fig.text(-.02, 0.5, ylabel, va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig('pictures_report/'+name+' '+str(I_noise)+'.png')
        
<<<<<<< HEAD
plot_all_histograms('burst', log = False)
=======
plot_all_histograms('up', log = False, cut = 50, fit = 'exp')
>>>>>>> 2aaf0206243d36e03c9603dc81dcafdc1f6422e6
