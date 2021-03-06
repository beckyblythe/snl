#import matplotlib.pyplot as plt
import argparse
from brian2 import *
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt


class Object(object):
    pass


plt.rcParams['figure.figsize'] = 9, 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#plt.rcParams['figure.fontsize'] = 16
#plt.rcParams['image.cmap'] = 'gray'


def get_simulation(file_name):
    '''reads simulation results from file or runs simulation'''
    #read from file
    with open('simulations/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
    return data_loaded        

def simulate_neuron(tau_n, Iapp, number, v0, n0, duration, I_noise, dt=.001*ms):
    '''runs simulation, returns M,Mv and Mn as objects with dimensionless np.arrays attributes'''
    #run simulation       
    neuron = NeuronGroup(number, eqs,  threshold = 'v >-.02*volt', refractory = 'v > -.03*volt')
    neuron.v = v0 
    neuron.n = n0
        
    M = SpikeMonitor(neuron, variables = 'v')
    Mv = StateMonitor(neuron, 'v', record=True)
    Mn = StateMonitor(neuron, 'n', record=True)
    
    defaultclock.dt = dt
    run(duration, report='text')

    return M.t_, M.i_, Mv.t_, Mv.v_, Mn.n_    
    
def find_points(tau_n, Iapp, v0=[-75,-32.3,-50]*mV,n0=[.0,0,.005], plot = False):
    '''finds the node and lowest point of limimt cycle in terms of voltage values
        trying to define threshold automatically '''
    dt=.00005*ms
    Spike_t, Spike_i, t, V, n = simulate_neuron(tau_n=tau_n, Iapp=Iapp, number = 3, v0=v0, n0=n0, duration=20*ms, 
                                      I_noise=0*uA, dt=dt)
    V*=1000
    node = [max(V[0]),n[0,np.argmax(V[0])]]
    cycle_boundary= [min (V[-1]), n[0,np.argmin(V[-1])]]
    saddle = find_saddle(tau_n, Iapp, node, cycle_boundary)
    sep_slope = find_sep_approx(tau_n, Iapp, saddle)
    
    file_name = str(tau_n)+'  '+str(Iapp)
    if plot:
        plot_bifurcation(t,V[-1],n[-1], node, saddle, sep_slope, cycle_boundary)
        plot_field(tau_n, Iapp, plot=True)
        cut = np.where(V[-2]>=saddle[0])
#        print(V[-2][cut])
#        plt.plot(V[-2][cut], n[-2][cut], color = 'grey', linewidth = 5)
        plt.savefig('points/'+file_name+'.png', bbox_inches = 'tight') 
        plt.show()

    return node, saddle, sep_slope, cycle_boundary
    
def get_points(tau_n, Iapp, plot=False):
    '''loads or calculates node, saddle and limit cycle lowest voltage '''
    file_name = 'points/'+str(tau_n)+'  '+str(Iapp)
    #read from file
    try:
        with open(file_name, 'rb') as f:
            data_loaded = pickle.load(f)
        node = data_loaded['node']
        saddle = data_loaded['saddle']
        sep_slope = data_loaded['sep_slope']
        cycle_boundary = data_loaded['cycle_boundary']
    except FileNotFoundError:
        node, saddle, sep_slope, cycle_boundary=find_points(tau_n, Iapp, plot=plot)
        points_calculated = {'node':node, 'saddle':saddle, 'sep_slope':sep_slope, 'cycle_boundary':cycle_boundary}
        with open(file_name, 'wb') as f:
            pickle.dump(points_calculated, f)   
            
    return node, saddle, sep_slope, cycle_boundary
           
def plot_everything(tau_n, Iapp, duration, I_noise, number =1, plot = False):
    '''simulates neuron and plots all the available plots'''
    file_name=str(duration/second)+' s  '+str(I_noise)+' uA  '+str(tau_n)+'  '+str(Iapp)+' '+str(number)
    print(file_name)
    
    node, saddle, sep_slope, cycle_boundary = get_points(tau_n, Iapp, plot) 
    if node[0] >= cycle_boundary[0]-.00001:
        print('We are not in the bistable regime! Consider trying different parameters!.')
        
    try:     
        results = get_simulation(file_name)
        print('Other plots are already generated. Find them in traces folder.')
    
    except IOError:
        v0=(np.ones(number)*saddle[0]+np.random.randn(number))*mV
        n0=np.ones(number)*saddle[1]+np.random.randn(number)*.01
        Spike_t, Spike_i, t, V, n= simulate_neuron(tau_n, Iapp, number, v0, n0, duration, I_noise)
        V*=1000
       
        keys = ['quiet_ISI_indices', 'quiet_ISI_start', 'quiet_ISI_end', 'break_point', 
                'first_down', 'last_down', 'first_up', 'last_up', 'ISI', 'ISI_quiet', 
                'ISI_burst', 'time_above', 'time_down','time_up', 'n_first_down', 
                'n_last_down', 'n_first_up', 'n_last_up']
        results = {key:[] for key in keys}
        
        for i in range(number):
            V_i=V[i]
            n_i=n[i]
            Spikes = Spike_t[np.where(Spike_i == i)[0]]
            result_i = collect_ISI_stats(t, V_i, n_i, Spikes, saddle, sep_slope, node)
            for key in keys:
                results[key].append(result_i[key]) 
        
        with open('simulations/'+file_name, 'wb') as f:
            pickle.dump(results, f) 
            
        if plot and duration <= 100*ms:
            plot_traces(t,V,n,node, saddle, sep_slope, cycle_boundary)
            plt.savefig('traces/'+file_name+'.png')
                    
    if plot:
        plot_histograms(results) 
            
        if duration/ms > 1000:
            plt.savefig('histograms/'+file_name+'.png')
        plt.show()
   
  
def plot_bifurcation(t,V,n,node,saddle, sep_slope, cycle_boundary):
    plt.figure(figsize=(3, 3))
    plt.plot(V.T,n.T, color = '#4B0082', linewidth = 3)
    plt.plot(node[0], node[1],marker='o', color='0', ms = 10)
#    y = np.linspace(-.1,.7,50)
#    x = sep_slope[0]/sep_slope[1]*(y-saddle[1])+saddle[0]
#    plt.plot(x,y, color = '0', linestyle = '--',linewidth = 5)
    plt.xlim((-70,0))
    plt.plot(saddle[0], saddle[1], marker = 'o', color = '.5', ms=10)
    plt.xticks()
    plt.ylim((-.05,.7))
    plt.yticks()
#    plt.title(' ', fontsize = 16)
#    plt.xlabel('V (mV)')
#    plt.ylabel('n')
#    plt.tight_layout()
    
def plot_traces(t,V,n,node, saddle, sep_slope, cycle_boundary):
    '''plots voltage against time'''   

    plt.subplot2grid((2,2),(0,0), colspan=2)
    plt.title('Voltage trace', fontsize = 16)
    plt.xlabel('time (ms)', fontsize = 16)
    plt.ylabel('voltage (mV)', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.plot(t*1000, V.T) 
    plt.axhline(y = node[0],color='0', linestyle ='--')
    plt.axhline(y = saddle[0],color='.5', linestyle ='--')
    plt.subplot2grid((2,2),(1,0))
    plt.title('Trajectory in V-n plane', fontsize = 16)
    plt.xlabel('voltage (mV)', fontsize = 16)
    plt.ylabel('n', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.plot(V.T,n.T)
    plt.plot(node[0], node[1],marker='o', color='0')
    plt.plot(saddle[0], saddle[1], marker = 'o', color = '.5')
    y = np.linspace(-.1,.7,50)
    x = sep_slope[0]/sep_slope[1]*(y-saddle[1])+saddle[0]
    plt.plot(saddle[0], saddle[1], color = '0')
    plt.plot(x,y, color = '0', linestyle = '--',linewidth = 2)
    plt.xlim((-70,0))
    plt.ylim((-.05,.7))
#    plot_field(tau_n, Iapp, plot = True)
    plt.subplot2grid((2,2),(1,1))
    plt.title('Trajectory in V-n plane (zoomed)', fontsize = 16)
    plt.xlabel('voltage (mV)', fontsize = 16)
    plt.ylabel('n', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.plot(V.T,n.T)
    plt.plot(node[0], node[1],marker='o', color='0')
    plt.plot(saddle[0], saddle[1], marker = 'o', color = '.5')
    plt.xlim((min(node[0]-(cycle_boundary[0]-node[0]),cycle_boundary[0]+(cycle_boundary[0]-node[0])), 
              max(node[0]-3.5*(cycle_boundary[0]-node[0]),cycle_boundary[0]+3.5*(cycle_boundary[0]-node[0]))))
    y = np.linspace(-.05,.5,50)
    x = sep_slope[0]/sep_slope[1]*(y-saddle[1])+saddle[0]
    plt.plot(x,y, color = '0', linestyle = '--',linewidth = 2)
    plt.ylim((-.01,.4))
    plt.tight_layout()  

    
def quiet_stats(t, V, n, Spikes, saddle, sep_slope,node):
    '''We count as quiet ISIs when V reached the neighbourhhod of the node'''
    #array of booleans, length = length of t array
    below_sep = ((V-saddle[0])/sep_slope[0]-(n-saddle[1])/sep_slope[1]<=0) & (t>Spikes[0]) & (t < Spikes[-1])
    above_sep = np.invert(below_sep)
    around_node = (V<=node[0]+.1*(saddle[0]-node[0])) & (t>Spikes[0]) & (t < Spikes[-1])
    #array of spikes indices, after which there is a quiet ISI
    quiet_ISI_indices = np.unique(np.searchsorted(Spikes, t[around_node]))-1
    print("1")
    t_around_node = t[around_node]
    t_below_sep = t[below_sep]
    t_above_sep = t[above_sep]
    n_below_sep = n[below_sep]
    n_above_sep = n[above_sep]
    print("2")
    #check this again   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 
    quiet_ISI_start = Spikes[quiet_ISI_indices]
    quiet_ISI_end = Spikes[quiet_ISI_indices+1]
    break_point = t_around_node[np.unique(np.searchsorted(t_around_node,quiet_ISI_start))]
    first_below_sep_indices = np.unique(np.searchsorted(t_below_sep, quiet_ISI_start))
    first_above_sep_indices = np.unique(np.searchsorted(t_above_sep, break_point))
    last_above_sep_indices = first_above_sep_indices-1
    last_below_sep_indices = np.unique(np.searchsorted(t_below_sep, quiet_ISI_end))-1
    print("3")
    first_down = t_below_sep[first_below_sep_indices]
    last_down = t_above_sep[last_above_sep_indices]
    first_up = t_above_sep[first_above_sep_indices]
    last_up = t_below_sep[last_below_sep_indices]
    print("4")
    n_first_down = n_below_sep[first_below_sep_indices]
    n_last_down = n_above_sep[last_above_sep_indices]
    n_first_up = n_above_sep[first_above_sep_indices]
    n_last_up = n_below_sep[last_below_sep_indices] 
    #also do n_first_up here!!!!!!
#    print(n_last_up)
    print("5")
    
    return quiet_ISI_indices, quiet_ISI_start, quiet_ISI_end, break_point, first_down, last_down, first_up, last_up, n_first_down, n_last_down, n_first_up, n_last_up
    
def calculate_quiet_ISIs_partition(ISI_quiet, break_point, last_down, last_up):
    '''Calculates quiet ISIs segments'''
    time_above = ISI_quiet - (last_up-last_down)
    time_down = break_point-last_down
    time_up = last_up-break_point
   
    return time_above, time_down, time_up
    
def collect_ISI_stats(t, V,n, Spikes, saddle, sep_slope, node):
    '''Calculates all ISIs, classifies ISIs, and partiotions quiet ISIs into segments'''
    quiet_ISI_indices, quiet_ISI_start, quiet_ISI_end, break_point, first_down, last_down, first_up, last_up, n_first_down, n_last_down, n_first_up, n_last_up = quiet_stats(t, V, n, Spikes, saddle, sep_slope, node)
    ISI = calculate_ISI(Spikes)
    ISI_quiet = ISI[quiet_ISI_indices]
    ISI_burst = np.delete(ISI,quiet_ISI_indices)
    time_above, time_down, time_up = calculate_quiet_ISIs_partition(ISI_quiet, break_point, last_down, last_up)
    result = locals()
    return result
                 

def calculate_ISI(Spikes):
    '''calculates all ISIs lengths'''
    return np.diff(Spikes)
    
def plot_histograms(results):
    '''plots histogram for all ISIs, classified ISIs, and quiet ISIs segments'''
    
    fig = plt.figure()
    keys_ordered = ['ISI', 'ISI_quiet', 'ISI_burst','time_down',
                'time_up', 'time_above']
    intervals = results
    for i, key in enumerate(keys_ordered):
        flat_intervals = np.array([interval for neuron in intervals[key] for interval in neuron])
           
        ax = fig.add_subplot(2,3,i+1)
        ax.set_title(str(flat_intervals.shape[0])+ ' ' + str(key))
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Distribution of times')
        if key == 'ISI':
            cut = np.where(flat_intervals <= 1*np.median(flat_intervals))
            ax.hist(flat_intervals[cut]*1000, normed = True, bins = 50, log = True)
        else:
            ax.hist(flat_intervals*1000, normed = False, bins = 50)
#        ax.axvline(flat_intervals.mean()*1000, color = 'r')
        ax.axvline(np.median(flat_intervals)*1000, color = 'b')
        if key == 'ISI': 
#            ax.set_xlim((0,5*flat_intervals.mean()*1000))
            xmin,xmax = ax.get_xlim()
        elif key in ['ISI_burst', 'time_down', 'time_above']:
            ax.set_xlim((0,10))
        else:
            ax.set_xlim=((xmin,xmax))
    
    plt.tight_layout()  
    
def plot_quiet_and_burst_hist(results):
    intervals = results
    keys_ordered = ['ISI_quiet', 'ISI_burst']
    colors = ['b','r']
    plt.figure()
    flat_intervals = np.array([interval for neuron in intervals['ISI'] for interval in neuron])
    mean = flat_intervals.mean()*1000
    for key, color in zip(keys_ordered,colors):
        flat_intervals = np.array([interval for neuron in intervals[key] for interval in neuron])
        plt.hist(flat_intervals*1000, normed = False, bins = 50, color=color, alpha = .5, range = ((0,10*mean)), log = False)
    plt.tight_layout()
    
def plot_segments_hist(results):
    intervals = results
    keys_ordered = ['time_above','time_up', 'time_down']
    colors = ['b','r', 'g']
    plt.figure()
    flat_intervals = np.array([interval for neuron in intervals['ISI'] for interval in neuron])
    mean = flat_intervals.mean()*1000
    for key, color in zip(keys_ordered,colors):
        flat_intervals = np.array([interval for neuron in intervals[key] for interval in neuron])
        plt.hist(flat_intervals*1000, normed = True, bins = 50, color=color, alpha = .5, range = ((0,10*mean)), log = False)
    plt.tight_layout()
    
def find_saddle(tau_n, Iapp, node, cycle_boundary):
    '''Finds saddle location given parameters'''
    #Find point with smallest derivative vector located between the node and the limit cycle
    v_grid ,n_grid = np.meshgrid(np.linspace(node[0]+.05*(cycle_boundary[0]-node[0]),cycle_boundary[0],5000), 
                                 np.linspace(-0,.05,1000))
    dv_grid = ((-g_Na*1./(1+exp((-20-v_grid)/15.))*(v_grid-E_Na/mV)-g_K*n_grid*(v_grid-E_K/mV)
                -g_L*(v_grid-E_L/mV)+Iapp/mV)/Cm)*ms
    dn_grid = (1./(1+exp((-25-v_grid)/5.))-n_grid)/tau_n*ms
    norm =  np.sqrt(np.square(dv_grid)+np.square(dn_grid))
    saddle = [v_grid[np.unravel_index(norm.argmin(), norm.shape)],
              n_grid[np.unravel_index(norm.argmin(), norm.shape)]]
   
    return saddle
    
def find_sep_approx(tau_n, Iapp, saddle):
    '''Approximates separatrix with a straight line coming from saddle 
    in the direction of eigenvector, corresponding to negative eigenvalue'''
    #Calculate Jacobian
    a= 1/Cm*(-g_Na*(1000/15*exp((-20-saddle[0])/15)*(1+exp((-20-saddle[0])/15))**(-2)*(saddle[0]/1000-E_Na/volt)
                    + (1+exp((-20-saddle[0])/15))**(-1))- g_K*saddle[1]-g_L)*ms
    b= -1/Cm*g_K*(saddle[0]/1000-E_K/volt)*ms*1000
    c= 1/tau_n*1000/5*exp((-25-saddle[0])/5)*(1+exp((-25-saddle[0])/5))**(-2)*ms/1000
    d= -1/tau_n*ms
    J = np.array([[a,b],[c,d]])  
    #Find line direction
    eigval, eigvec = np.linalg.eig(J)
    sep_slope = eigvec[:,argmin(eigval)]
  
    return sep_slope
    
def plot_field(tau_n, Iapp, plot = False):
    '''Plots phase plane for given parameters'''
    v_grid ,n_grid = np.meshgrid(np.linspace(-70,0,8), np.linspace(-0,.7,8))
    dv_grid = ((-g_Na*1./(1+exp((-20-v_grid)/15.))*(v_grid-E_Na/mV)-g_K*n_grid*(v_grid-E_K/mV)
                -g_L*(v_grid-E_L/mV)+Iapp/mV/1000)/Cm)*ms/70
    dn_grid = (1./(1+exp((-25-v_grid)/5.))-n_grid)/tau_n*ms/.75
    #Normalization to have all vectors of the same length (otherwise the small ones are too small)
    norm = np.sqrt(np.square(dv_grid)+np.square(dn_grid))
    
#    print(dv_grid)
    dv_grid += 2*dv_grid/norm
    dn_grid += 2*dn_grid/norm
    
    if plot:
#        plt.figure()
        plt.quiver(v_grid,n_grid,dv_grid,dn_grid, width = .002)
#        plt.close()
#        plt.show()

###########################################
Cm = 1 * uF #/cm2
g_L = 8 * msiemens #/cm2
g_Na = 20 * msiemens#/cm2
g_K = 10 * msiemens #/cm2
E_L = -80 * mV
E_Na = 60 * mV
E_K = -90 * mV
tau = 1.0*ms
############################################
eqs = '''
dv/dt = (-I_Na - I_K -  I_L + Iapp+I_noise*sqrt(tau)*xi)/Cm : volt
dn/dt = (n_inf-n)/tau_n : 1
I_Na = g_Na*m_inf*(v-E_Na) : amp
I_K = g_K*n*(v-E_K) : amp
I_L = g_L*(v-E_L) : amp
n_inf = 1./(1+exp((-25-v/mV)/5.)) : 1
m_inf = 1./(1+exp((-20-v/mV)/15.)) : 1
'''
############################################

#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('parameters', metavar='N', type=float, nargs='+',
#                   help='tau_n, Iapp')
#args = parser.parse_args().parameters
#
##parameters to play with
#
#print ('tau_n = ' + str(args[0]))
#print ('Iapp = ' + str(args[1]))
#
#tau_n = args[0] * ms
#Iapp = args[1] * uA #/cm**2
tau_n = .155 * ms
Iapp =3 * uA #/cm**2
I_noise = 3*uA
duration = 40*ms

find_points(tau_n, Iapp, plot = True)

#plot_everything(tau_n=tau_n, Iapp=Iapp, duration=duration, I_noise=I_noise, number =5, plot=True)


#Spikes, t, V, n = simulate_neuron(tau_n, Iapp, 1, -30*mV, 0, duration, I_noise)
#ISIs = calculate_ISI(Spikes)
#plt.hist(ISIs, bins = 100)
#find_points(tau_n=tau_n, Iapp=Iapp, plot = True)
#find_sep_approx(tau_n=tau_n, Iapp=Iapp)
#plot_field(tau_n, Iapp, plot = True)
