from brian2 import *
import numpy as np
import pickle


class Object(object):
    pass


plt.rcParams['figure.figsize'] = 12, 4
plt.rcParams['agg.path.chunksize'] = 10000



def get_simulation(file_name):
    '''reads simulation results from file or runs simulation'''
    #read from file
    with open('old_model/' + 'simulations/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
    return data_loaded        

def simulate_neuron(Cm, Iapp, number, v0, n0, duration, I_noise,h0=0):
    '''runs simulation, returns M,Mv and Mn as objects with dimensionless np.arrays attributes'''
    #run simulation    
    neuron = NeuronGroup(number, eqs,  threshold = 'v >-.01*volt', refractory = 'v > -.02*volt')
    neuron.v = v0 
    neuron.n = n0
    neuron.h = h0
    
    M_temp = SpikeMonitor(neuron, variables = 'v')
    Mv_temp = StateMonitor(neuron, 'v', record=True)
    Mn_temp = StateMonitor(neuron, 'n', record=True)
    
    run(duration, report='text')
        
    Spikes = np.array(M_temp.t)
    t = np.array(Mv_temp.t)
    V = np.array(Mv_temp.v)*1000 #-----> mV
    n = np.array(Mn_temp.n)
               
    return Spikes, t, V, n    
    
def find_points(Cm, Iapp, v0=[-70,-62,-60.5,-59.5,-58,-50]*mV,n0=[.3,.3,.3, 0,0,.3]):
    '''finds the node and lowest point of limimt cycle in terms of voltage values
        trying to define threshold automatically '''
    Spikes, t, V, n = simulate_neuron(Cm=Cm, Iapp=Iapp, number = 6, v0=v0, n0=n0, duration=2000*ms, I_noise=0*uA)
    node = max(V[0])
    cycle_boundary= min (V[-1])
    
    file_name = str(Cm)+'  '+str(Iapp)
    
    plot_traces(t,V,n, node, cycle_boundary)
    plt.savefig('old_model/' + 'points/'+file_name+'.png') 
    plt.show()
    
    if node >= cycle_boundary:
        raise Exception('We passed saddle-node bifurcation point or limit cycle doesnt exist! Try different parameters.')
    return node, cycle_boundary
    
def get_points(file_name):
    #read from file
    with open( 'old_model'+'points/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
    node = data_loaded['node']
    cycle_boundary = data_loaded['cycle_boundary']
    
    return node, cycle_boundary

def set_thresh(Cm, Iapp, weight=.5):
    file_name = str(Cm)+'  '+str(Iapp)
    
    try:
        node, cycle_boundary = get_points(file_name)
        print('Reading node and cycle boundary location from file.')
    except IOError:
        node, cycle_boundary = find_points(Cm=Cm, Iapp=Iapp)
    #setting threshold in the middle between the node and the limit cycle
        data_generated = {'node':node,'cycle_boundary':cycle_boundary}
        with open('old_model/' + 'points/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f) 
        
    thresh = weight*node+(1-weight)*cycle_boundary

    return thresh, node, cycle_boundary
        
    
def plot_everything(Cm, Iapp, duration, I_noise, weight, number =1, v0=-50*mV, n0=0):
    '''simulates neuron and plots all the available plots'''
    
    thresh,node,cycle_boundary = set_thresh(Cm, Iapp, weight)
    
    file_name=str(Cm)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(int(duration/second))+' s  '+str(I_noise) + ' '+str(weight)
    print(file_name)
    try:     
        data= get_simulation(file_name)
        print('Other plots are already generated. Find them in traces folder.')
        plot_histograms(node, **data)
        if duration/ms >=50000:
            plt.savefig('old_model/' + 'histograms/'+file_name+'.png')
        plt.show()
        
    except IOError:
        Spikes, t, V, n= simulate_neuron(Cm, Iapp, number, v0, n0, duration, I_noise)
        
                
        plot_traces(t,V,n,node, cycle_boundary)
        plt.savefig('old_model/' + 'traces/'+file_name+'.png') 
        plt.show()
        
        V=V[-1]
        ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up = collect_ISI_stats(t, V, Spikes, thresh, node)
        plot_histograms(node, ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up) 
        if duration/ms >=50000:
            plt.savefig('old_model/' + 'histograms/'+file_name+'.png')
        plt.show()
        
        #to do with a loop
        data_generated =  {'Min_Volt':Min_Volt,'ISI':ISI, 'ISI_quiet':ISI_quiet, 'ISI_burst':ISI_burst, 'time_above':time_above, 'time_down':time_down, 'time_up':time_up }
        with open('old_model/' + 'simulations/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f)   
            
   
  
def plot_traces(t,V,n,node, cycle_boundary):
    '''plots voltage against time'''   
    plt.figure(figsize=(12, 8))
    plt.subplot2grid((2,2),(0,0), colspan=2)
    plt.title('Voltage trace')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.plot(t, V.T) 
    plt.axhline(y=node, linestyle = ':',color = 'm')
    plt.subplot2grid((2,2),(1,0))
    plt.title('Trajectory in V-n plane')
    plt.xlabel('voltage (mV)')
    plt.ylabel('n')
    plt.plot(V.T,n.T)
    plt.subplot2grid((2,2),(1,1))
    plt.title('Trajectory in V-n plane (zoomed)')
    plt.xlabel('voltage (mV)')
    plt.ylabel('n')
    plt.plot(V.T,n.T)
    plt.axvline(x=node, linestyle = ':', color='m')
    plt.axvline(x=cycle_boundary,linestyle = ':', color='c')
    plt.xlim((min(node-2*(cycle_boundary-node),cycle_boundary+2*(cycle_boundary-node)), 
              max(node-2*(cycle_boundary-node),cycle_boundary+2*(cycle_boundary-node))))
    plt.ylim((-.1,.5))
    
def quiet_stats(t, V, Spikes, thresh,node):
    '''Now we count as quiet ISIs where V reached the node value'''
    #array of booleans, length = length of t array
    below_thresh = (V<=thresh) & (t>Spikes[0]) & (t < Spikes[-1])
    below_node = (V<=node) & (t>Spikes[0]) & (t < Spikes[-1])
    #array of spikes indices, after which there is a quiet ISI
    quiet_ISI_indices = np.unique(np.searchsorted(Spikes, t[below_node]))-1
    #indices of t elements corresponding to Spike times
    spike_times_indices = np.searchsorted(t, Spikes,side='left')
    
    Min_Volt = np.empty(quiet_ISI_indices.shape[0])
    break_point = np.empty(quiet_ISI_indices.shape[0])
    Crossing_down = np.empty(quiet_ISI_indices.shape[0])
    Crossing_up = np.empty(quiet_ISI_indices.shape[0])
    
    for i in range(quiet_ISI_indices.shape[0]):  
        #array of t indices within each quiet ISI
        Slice=np.arange(spike_times_indices[quiet_ISI_indices[i]],spike_times_indices[quiet_ISI_indices[i]+1])
        Min_Volt[i] = np.min(V[Slice])
#        print(np.nonzero(V[Slice]<=node))
        break_point[i]    = t[Slice[0]+np.min(np.nonzero(V[Slice]<=node))]  
        Crossing_down[i] = t[Slice[0] + np.min(np.nonzero(below_thresh[Slice]))]
        Crossing_up[i] = t[Slice[0] + np.max(np.nonzero(below_thresh[Slice]))]
    return quiet_ISI_indices, Min_Volt, break_point, Crossing_down, Crossing_up 
    
def calculate_quiet_ISIs_partition(ISI_quiet, break_point, Crossing_down, Crossing_up):
    time_above = ISI_quiet - (Crossing_up-Crossing_down)
    time_down = break_point-Crossing_down
    time_up = Crossing_up-break_point
   
    return time_above, time_down, time_up
    
def collect_ISI_stats(t, V, Spikes, thresh, node):
    quiet_ISI_indices, Min_Volt, break_point, Crossing_down, Crossing_up = quiet_stats(t, V, Spikes, thresh, node)
    ISI = calculate_ISI(Spikes)
    ISI_quiet = ISI[quiet_ISI_indices]
    ISI_burst = np.delete(ISI,quiet_ISI_indices)
    time_above, time_down, time_up = calculate_quiet_ISIs_partition(ISI_quiet, break_point, Crossing_down, Crossing_up)
    return ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up
                    

def calculate_ISI(Spikes):
    '''calculates all ISIs lengths'''
    return np.diff(Spikes)
    
def plot_histograms(node, ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up):
    '''plots histogram for ISIs and classified ISIs'''
    
    plt.figure(figsize = (12,8))
    
    plt.subplot(2,3,1)
    plt.title('All '+ str(ISI.shape[0]) + ' ISIs. ')
    plt.xlabel('ISI (s)')
    plt.ylabel('Distribution of ISIs')
    plt.xlim((0,5))
    plt.hist(ISI, normed = True, bins = 50)
    plt.axvline(ISI.mean(), color = 'r')
    plt.subplot(2,3,2)
    plt.title(str(ISI_quiet.shape[0]) + ' Quiet ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_quiet, normed = True, bins = 50)
    plt.axvline(ISI_quiet.mean(), color = 'r')
    plt.xlim((0,5))
    plt.subplot(2,3,3)
    plt.title(str(ISI_burst.shape[0]) + ' Burst ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_burst,normed = True, bins = 50)
    plt.axvline(ISI_burst.mean(), color = 'r')
    plt.xlim((0,1))
    plt.subplot(2,3,4)
    plt.title('Time from thresh to node')
    plt.hist(time_down, normed = True)
    plt.axvline(time_down.mean(), color = 'r')
    plt.xlabel('time (s)')
    plt.ylabel('Distribution of times')
    plt.xlim((0,1))
    plt.subplot(2,3,5)
    plt.title('Time from node to thresh')
    plt.hist(time_up, normed = True)
    plt.axvline(time_up.mean(), color = 'r')
    plt.xlabel('time (s)')
    plt.xlim((0,5))
    plt.subplot(2,3,6)
    plt.title('Time above the thresh')
    plt.hist(time_above, normed = True)
    plt.axvline(time_above.mean(), color = 'r')
    plt.xlabel('time (s)')
    plt.xlim((0,1))
    plt.tight_layout()  
    
defaultclock.dt = 0.05*ms


gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens
tau=1*ms

Cm = 1.65*uF # /cm**2
Iapp = 0.160*uA
I_noise = .100*uA
duration = 5000000*ms

weight=.95 #after data is saved we can't change the weight anymore

eqs = '''
dv/dt = (-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp+I_noise*sqrt(tau)*xi)/Cm : volt
m = alpha_m/(alpha_m+beta_m) : 1
alpha_m = -0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
alpha_n = -0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
'''


#find_points(Cm = Cm, Iapp = Iapp)
plot_everything(Cm, Iapp, duration, I_noise, weight, v0=-50*mV, n0=0)