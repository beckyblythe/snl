from brian2 import *
import numpy as np
import pickle


class Object(object):
    pass


plt.rcParams['figure.figsize'] = 12, 4

defaultclock.dt = 0.05*ms


gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens
tau=1*ms

Cm = 1.65*uF # /cm**2
Iapp = .158*uA
I_noise = .07*uA
duration = 10000*ms

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

def get_simulation(file_name):
    '''reads simulation results from file or runs simulation'''
    #read from file
    with open('simulations/'+file_name, 'rb') as f:
        data_loaded = pickle.load(f)
        
    Spikes=data_loaded['Spikes']
    Min_Volt=data_loaded['Min_Volt']
     
    return Spikes, Min_Volt        

def simulate_neuron(Cm, Iapp, number, v0, n0, duration, I_noise):
    '''runs simulation, returns M,Mv and Mn as objects with dimensionless np.arrays attributes'''
    #run simulation    
    neuron = NeuronGroup(number, eqs,  threshold = 'v >-.01*volt', refractory = 'v > -.01*volt')
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
    
def find_points(Cm=Cm, Iapp=Iapp):
    '''finds the node and lowest point of limimt cycle in terms of voltage values
        trying to define threshold automatically '''
    Spikes, t, V, n = simulate_neuron(Cm=Cm, Iapp=Iapp, number = 2, v0=[-100, 0]*mV, n0=[.1,.1], duration=1000*ms, I_noise=0*uA)
    node = max(V[0])
    cycle_boundary= min (V[1])
      
    return node, cycle_boundary
    
def get_points(file_name):
    #read from file
    with open('points/'+file_name, 'rb') as f:
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
        with open('points/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f) 
        
    thresh = weight*node+(1-weight)*cycle_boundary

    return thresh, node, cycle_boundary
        
    
def plot_everything(Cm=Cm, Iapp=Iapp, number =1, v0=-55*mV, n0=.1, duration=duration, I_noise=I_noise, weight=.5):
    '''simulates neuron and plots all the available plots'''
    
    thresh,node,cycle_boundary = set_thresh(Cm, Iapp, weight)
    
    file_name=str(Cm)+'  '+str(Iapp)+'  ('+str(v0)+', '+str(n0)+')  '+str(int(duration/second))+' s  '+str(I_noise)
      
    try:     
        Spikes, Min_Volt = get_simulation(file_name)
        print('Other plots are already generated. Find them in traces folder.')
        plot_histograms(Spikes, Min_Volt, thresh)
        plt.savefig('histograms/'+file_name+'  '+str(weight)+'.png')
        plt.show()
        
    except IOError:
        Spikes, t, V, n= simulate_neuron(Cm, Iapp, number, v0, n0, duration, I_noise)
        Min_Volt= find_period_minima(V.flatten(),t, Spikes)
                
        plot_traces(t,V,n,node, cycle_boundary)
        plt.savefig('traces/'+file_name+'.png') 
        plt.show()
        plot_histograms(Spikes,Min_Volt,thresh) 
        plt.savefig('histograms/'+file_name+'  '+str(weight)+'.png')
        plt.show()
        
        data_generated =  {'Spikes':Spikes,'Min_Volt':Min_Volt}
        with open('simulations/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f)        
  
def plot_traces(t,V,n,node=0, cycle_boundary=0):
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
    plt.xlim((node-3*(cycle_boundary-node),cycle_boundary+3*(cycle_boundary-node)))
    plt.ylim((-.1,.5))
    
            
def find_period_minima(timecourse, time, section_array):
    print()
    section_indices = np.where(np.in1d(time, section_array))[0]
    minima = np.empty(section_indices.shape[0]-1)
    for i in range(section_indices.shape[0]-1):
        minima[i] = np.min(timecourse[section_indices[i]:section_indices[i+1]])
    return minima        
    
def classify_ISI_indices(Min_Volt, thresh =-59.5):
    '''returns tuple: indices of spikes after which there is a quiet interval,
                      indices of spikes after which there is a burst interval '''
    indices_quiet = np.where(Min_Volt < thresh)
    indices_burst = np.where(Min_Volt >= thresh)
    return indices_quiet, indices_burst

def calculate_ISI(Spikes):
    '''calculates all ISIs lengths'''
    return np.diff(Spikes)
    
def classify_ISI_lengths(Spikes, Min_Volt, thresh =-59.5):
    '''calculates ISI lengths and classifies them into quiet and burst'''
    indices_quiet, indices_burst = classify_ISI_indices(Min_Volt, thresh)
    ISI = calculate_ISI(Spikes)
    ISI_quiet = ISI[indices_quiet]
    ISI_burst = ISI[indices_burst]
    return ISI, ISI_quiet, ISI_burst
    
def plot_histograms(Spikes, Min_Volt, thresh =-59.5):
    '''plots histogram for ISIs and classified ISIs'''
    ISI, ISI_quiet, ISI_burst = classify_ISI_lengths(Spikes, Min_Volt, thresh)
        
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('All '+ str(ISI.shape[0]) + ' ISIs. ')
    plt.xlabel('ISI (s)')
    plt.hist(ISI)
    plt.subplot(1,3,2)
    plt.title(str(ISI_quiet.shape[0]) + ' Quiet ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_quiet)
    plt.subplot(1,3,3)
    plt.title(str(ISI_burst.shape[0]) + ' Burst ISIs')
    plt.xlabel('ISI (s)')
    plt.hist(ISI_burst)
    

plot_everything(weight = 0)

        

    


