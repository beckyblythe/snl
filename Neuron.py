from brian2 import *
import numpy as np
import pickle


class Object(object):
    pass


plt.rcParams['figure.figsize'] = 6, 6
plt.rcParams['agg.path.chunksize'] = 10000



def get_simulation(file_name):
    '''reads simulation results from file or runs simulation'''
    #read from file
    with open('simulations/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
    return data_loaded        

def simulate_neuron(tau_n, Iapp, number, v0, n0, duration, I_noise):
    '''runs simulation, returns M,Mv and Mn as objects with dimensionless np.arrays attributes'''
    #run simulation    
    print(tau_n, Iapp)
    neuron = NeuronGroup(number, eqs,  threshold = 'v >-.03*volt', refractory = 'v > -.03*volt')
    neuron.v = v0 
    neuron.n = n0
    
    
    M_temp = SpikeMonitor(neuron, variables = 'v')
    Mv_temp = StateMonitor(neuron, 'v', record=True)
    Mn_temp = StateMonitor(neuron, 'n', record=True)
    
    run(duration, report='text')
        
    Spikes = np.array(M_temp.t)
    t = np.array(Mv_temp.t)
    V = np.array(Mv_temp.v)*1000 #-----> mV
    n = np.array(Mn_temp.n)
               
    return Spikes, t, V, n    
    
def find_points(tau_n, Iapp, v0=[-75,-65,-40, -65,-60,-50]*mV,n0=[.05,.05,-.1, -.1,-.1,-0]):
    '''finds the node and lowest point of limimt cycle in terms of voltage values
        trying to define threshold automatically '''
    Spikes, t, V, n = simulate_neuron(tau_n=tau_n, Iapp=Iapp, number = 6, v0=v0, n0=n0, duration=1000*ms, I_noise=0*uA)
    node = max(V[0])
    cycle_boundary= min (V[-1])
    
    file_name = str(tau_n)+'  '+str(Iapp)
    
    plot_traces(t,V,n, node, cycle_boundary)
    plt.savefig('points/'+file_name+'.png') 
    plt.show()
    
    if node >= cycle_boundary:
        raise Exception('We passed saddle-node bifurcation point or limit cycle doesnt exist! Try different parameters.')
    return node, cycle_boundary
    
def get_points(tau_n, Iapp):
    file_name = str(tau_n)+'  '+str(Iapp)
    #read from file
    with open('points/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
    node = data_loaded['node']
    cycle_boundary = data_loaded['cycle_boundary']
    
    return node, cycle_boundary

def set_thresh(tau_n, Iapp, weight=.5):
    
    try:
        node, cycle_boundary = get_points(file_name)
        print('Reading node and cycle boundary location from file.')
    except IOError:
        file_name = str(tau_n)+'  '+str(Iapp)
        node, cycle_boundary = find_points(tau_n=tau_n, Iapp=Iapp)
    #setting threshold in the middle between the node and the limit cycle
        data_generated = {'node':node,'cycle_boundary':cycle_boundary}
        with open('points/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f) 
        
    thresh = weight*node+(1-weight)*cycle_boundary

    return thresh, node, cycle_boundary
        
    
def plot_everything(tau_n, Iapp, duration, I_noise, weight, number =1, v0=-30*mV, n0=-0):
    '''simulates neuron and plots all the available plots'''
    
    thresh,node,cycle_boundary = set_thresh(tau_n, Iapp, weight)
    
    file_name=str(tau_n)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(int(duration/second))+' s  '+str(I_noise) + ' '+str(weight)
    print(file_name)
    try:     
        data= get_simulation(file_name)
        print('Other plots are already generated. Find them in traces folder.')
        plot_histograms(node, **data)
        if duration/ms >=50000:
            plt.savefig('histograms/'+file_name+'.png')
        plt.show()
        
    except IOError:
        Spikes, t, V, n= simulate_neuron(tau_n, Iapp, number, v0, n0, duration, I_noise)
        
                
        plot_traces(t,V,n,node, cycle_boundary)
        plt.savefig('traces/'+file_name+'.png') 
        plt.show()
        
        V=V[-1]
        ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up = collect_ISI_stats(t, V, Spikes, thresh, node)
        plot_histograms(node, ISI, ISI_quiet, ISI_burst, Min_Volt, time_above, time_down, time_up) 
        if duration/ms >=10000:
            plt.savefig('histograms/'+file_name+'.png')
        plt.show()
        
        #to do with a loop
        data_generated =  {'Min_Volt':Min_Volt,'ISI':ISI, 'ISI_quiet':ISI_quiet, 'ISI_burst':ISI_burst, 'time_above':time_above, 'time_down':time_down, 'time_up':time_up }
        with open('simulations/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f)   
            
   
  
def plot_traces(t,V,n,node, cycle_boundary):
    '''plots voltage against time'''   
    plt.figure(figsize=(12, 8))
    plt.subplot2grid((2,2),(0,0), colspan=2)
    plt.title('Voltage trace')
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.plot(t, V.T) 
#    plt.axhline(y=node, linestyle = ':',color = 'm')
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
#    plt.xlim((0,5))
    plt.hist(ISI, normed = True)
    plt.axvline(ISI.mean(), color = 'r')
    plt.subplot(2,3,2)
    plt.title(str(ISI_quiet.shape[0]) + ' Quiet ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_quiet, normed = True)
    plt.axvline(ISI_quiet.mean(), color = 'r')
#    plt.xlim((0,5))
    plt.subplot(2,3,3)
    plt.title(str(ISI_burst.shape[0]) + ' Burst ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_burst,normed = True)
    plt.axvline(ISI_burst.mean(), color = 'r')
#    plt.xlim((0,1))
    plt.subplot(2,3,4)
    plt.title('Time from thresh to node')
    plt.hist(time_down, normed = True)
    plt.axvline(time_down.mean(), color = 'r')
    plt.xlabel('time (s)')
    plt.ylabel('Distribution of times')
#    plt.xlim((0,1))
    plt.subplot(2,3,5)
    plt.title('Time from node to thresh')
    plt.hist(time_up, normed = True)
    plt.axvline(time_up.mean(), color = 'r')
    plt.xlabel('time (s)')
#    plt.xlim((0,5))
    plt.subplot(2,3,6)
    plt.title('Time above the thresh')
    plt.hist(time_above, normed = True)
    plt.axvline(time_above.mean(), color = 'r')
    plt.xlabel('time (s)')
#    plt.xlim((0,1))
    plt.tight_layout()  
    
def plot_field(tau_n, Iapp):
#    print(tau_n, Iapp)
    v_grid ,n_grid = np.meshgrid(np.linspace(-70,-50,100), np.linspace(-0,.05,50))
    dv_grid = ((-g_Na*1./(1+exp((-20-v_grid)/15.))*(v_grid*mV-E_Na)-g_K*n_grid*(v_grid*mV-E_K)-g_L*(v_grid*mV-E_L)+Iapp)/Cm)/mV*ms
    dn_grid = (1./(1+exp((-25-v_grid)/5.))-n_grid)/tau_n*ms
    norm =  np.sqrt(np.square(dv_grid)+np.square(dn_grid))
    print(np.argmin(norm))
#    dv_grid= np.divide(dv_grid,norm)
#    dn_grid= np.divide(dn_grid,norm)
    
    plt.figure()
#    plt.axis('equal')
    plt.quiver(v_grid,n_grid,dv_grid,dn_grid, width = .0015)
    
def find_saddle(tau_n, Iapp):
    node, cycle_boundary=get_points(tau_n, Iapp)
    v_grid ,n_grid = np.meshgrid(np.linspace(node+.05*(cycle_boundary-node),cycle_boundary,50), np.linspace(-0,.05,50))
    dv_grid = ((-g_Na*1./(1+exp((-20-v_grid)/15.))*(v_grid*mV-E_Na)-g_K*n_grid*(v_grid*mV-E_K)-g_L*(v_grid*mV-E_L)+Iapp)/Cm)/mV*ms
    dn_grid = (1./(1+exp((-25-v_grid)/5.))-n_grid)/tau_n*ms
    norm =  np.sqrt(np.square(dv_grid)+np.square(dn_grid))
    saddle = (v_grid[np.unravel_index(norm.argmin(), norm.shape)],n_grid[np.unravel_index(norm.argmin(), norm.shape)])
    print(saddle)
    dv_grid= np.divide(dv_grid,norm)
    dn_grid= np.divide(dn_grid,norm)
    plt.figure()
    plt.quiver(v_grid,n_grid,dv_grid,dn_grid, width = .0015)
    return(saddle)
    
find_separatrix_lin(tau_n, Iapp):
    saddle = find_saddle(tau_n, Iapp)
    
    
    
defaultclock.dt = 0.001*ms


Cm = 1 * uF #/cm2
g_L = 8 * msiemens #/cm2
g_Na = 20 * msiemens#/cm2
g_K = 10 * msiemens #/cm2
E_L = -80 * mV
E_Na = 60 * mV
E_K = -90 * mV

tau = 1.0*ms

tau_n = .155*ms
Iapp = 2* uA #/cm**2
I_noise = 3*uA
duration = 1000*ms

weight=.5 #after data is saved we can't change the weight anymore

eqs = '''
dv/dt = (-I_Na - I_K -  I_L + Iapp+I_noise*sqrt(tau)*xi)/Cm : volt
dn/dt = (n_inf-n)/tau_n : 1

I_Na = g_Na*m_inf*(v-E_Na) : amp
I_K = g_K*n*(v-E_K) : amp
I_L = g_L*(v-E_L) : amp

n_inf = 1./(1+exp((-25-v/mV)/5.)) : 1
m_inf = 1./(1+exp((-20-v/mV)/15.)) : 1
'''

#plot_everything(tau_n=tau_n, Iapp=Iapp, duration=duration, I_noise=I_noise, weight=weight, number =1, v0=-30*mV, n0=-0)

find_saddle(tau_n=tau_n, Iapp=Iapp)
#plot_field(tau_n, Iapp)


#thresh, node, cycle_boundary = set_thresh(Cm, Iapp, weight)
#number=1000
#
#
#v0=np.ones(number)*node*mV
#h0=np.ones(number)*(0.1*(node+35)/(exp(-0.1*(node+35))-1))/((-0.1*(node+35)/(exp(-0.1*(node+35))-1))+(1./(exp(-0.1*(node+28))+1)))
#n0=np.ones(number)*(-0.01*(node+34)/(exp(-0.1*(node+34))-1))/((-0.01*(node+34)/(exp(-0.1*(node+34))-1))+(.125*(exp(-(node+44)/80))))
#duration=50000*ms
#
#Spikes, t, V, n = simulate_neuron(Cm, Iapp, number, v0, n0,duration , I_noise,h0)
#lines = np.arange(V.shape[0])
#
##plt.plot(V.T,n.T)
##plt.axvline(node)
##plt.axvline(cycle_boundary)
#
#plt.figure(figsize = (12,8))
#plt.subplot(2,3,1)
##lines = np.where(np.max(V[:,:int(V.shape[1]/6)], axis = 1)<=thresh)
#plt.hist(V[lines,int(V.shape[1]/6)].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#plt.subplot(2,3,2)
##lines = np.where(np.max(V[:,:int(V.shape[1]*2/6)], axis = 1)<=thresh)
#plt.hist(V[lines,int(V.shape[1]*2/6)].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#plt.subplot(2,3,3)
##lines = np.where(np.max(V[:,:int(V.shape[1]*3/6)], axis = 1)<=thresh)
#plt.hist(V[lines,int(V.shape[1]*3/6)].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#
#plt.subplot(2,3,4)
##lines = np.where(np.max(V[:,:int(V.shape[1]*4/6)], axis = 1)<=thresh)
#plt.hist(V[lines,int(V.shape[1]*4/6)].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#plt.subplot(2,3,5)
##lines = np.where(np.max(V[:,:int(V.shape[1]*5/6)], axis = 1)<=thresh)
#plt.hist(V[lines,int(V.shape[1]*5/6)].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#plt.subplot(2,3,6)
##lines = np.where(np.max(V, axis = 1)<=thresh)
#plt.hist(V[lines,-1].T, bins = 50)
#plt.axvline(node)
#plt.axvline(cycle_boundary)
#plt.xlim((-70,-50))
#
#plt.tight_layout() 
#plt.show()
        

    


