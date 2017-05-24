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
duration = 500000*ms

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

def get_simulation(Cm=Cm, Iapp=Iapp, number =1, v0=-55*mV, n0=.1, duration=duration, I_noise=I_noise):
    '''reads simulation results from file or runs simulation'''
    #file name includes all altering parameters
    file_name=str(Cm)+' '+str(Iapp)+' '+str(v0)+' '+str(n0)+' '+str(duration/second)+' s '+str(I_noise)
    #read from file if it exists
    try:
        with open('simulations/'+file_name, 'rb') as f:
            data_loaded = pickle.load(f)
            
        M=data_loaded['M']
        Mv=data_loaded['Mv']
        Mn=data_loaded['Mn']
    
    #run simulation if file doesn't exist and save results in the new file
    except IOError:
        M, Mv, Mn = simulate_neuron(Cm, Iapp, number, v0, n0, duration, I_noise)
        data_generated =  {'M':M,'Mv':Mv,'Mn':Mn}
        with open('simulations/'+file_name, 'wb') as f:
            pickle.dump(data_generated, f)
            
                
    return M, Mv, Mn
        

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
    
    #transform results (otherwise pickle doen't work)
    M=Object()
    Mv=Object()
    Mn=Object()
    
    M.t = np.array(M_temp.t)
    Mv.t = np.array(Mv_temp.t)
    Mv.v = np.array(Mv_temp.v)*1000 #-----> mV
    Mn.t = np.array(Mn_temp.t)
    Mn.n = np.array(Mn_temp.n)
               
    return M, Mv, Mn
    
    
def find_points(Cm=Cm, Iapp=Iapp):
    '''finds the node and lowest point of limimt cycle in terms of voltage values
        trying to define threshold automatically '''
    M, Mv, Mn = get_simulation(Cm=Cm, Iapp=Iapp, number = 2, v0=[-100, 0]*mV, n0=[.1,.1], duration=1000*ms, I_noise=0*uA)
    node = max(Mv.v[0])
    cycle_boundary= min (Mv.v[1])
    return node, cycle_boundary
    
def plot_everything(Cm=Cm, Iapp=Iapp, number =1, v0=-55*mV, n0=.1, duration=duration, I_noise=I_noise):
    '''simulates neuron and plots all the available plots'''
    node, cycle_boundary = find_points(Cm=Cm, Iapp=Iapp)
    #setting threshold in the middle between the node and the limit cycle
    thresh = .5*(node+cycle_boundary)
      
    M, Mv, Mn = get_simulation(Cm, Iapp, number, v0, n0, duration, I_noise)
       
    plot_voltage_trace(Mv)
    plot_trajectories(Mv,Mn, node, cycle_boundary)
    plot_histograms(M,Mv,thresh) 
    
    file_name=str(Cm)+' '+str(Iapp)+' '+str(v0)+' '+str(n0)+' '+str(duration/second)+' s '+str(I_noise)
    plt.savefig('histograms/'+file_name+'.png')

def plot_voltage_trace(Mv):
    '''plots voltage against time'''
    plt.figure()
    plt.title(str(Cm) + ' ' + str(Iapp)+' '+ str(I_noise))
    plt.plot(Mv.t, Mv.v.T)
    plt.show()

def plot_trajectories(Mv,Mn,node=0, cycle_boundary=0, xlim=(-65,-50)):
    '''plots trajectories of neurons in V-n plane'''
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(Mv.v.T,Mn.n.T)
    plt.subplot(1,2,2)
    plt.plot(Mv.v.T,Mn.n.T)
    plt.axvline(x=node)
    plt.axvline(x=cycle_boundary)
    plt.xlabel('mV')
    plt.ylabel('n')
    plt.xlim(xlim)
    plt.ylim((-0.1,1))
    plt.show()

def find_period_minima(timecourse, section_indices):
    minima = np.empty(section_indices.shape[0]-1)
    for i in range(section_indices.shape[0]-1):
        minima[i] = np.min(timecourse[section_indices[i]:section_indices[i+1]])
    return minima
        
    
def classify_ISI_indices(M,Mv, thresh =-59.5):
    '''returns tuple: indices of spikes after which there is a quiet interval,
                      indices of spikes after which there is a burst interval '''
    
    spike_indices = np.where(np.in1d(Mv.t, M.t))[0]
    voltage_minima = find_period_minima(Mv.v.flatten(),spike_indices)
    indices_quiet = np.where(voltage_minima < thresh)
    indices_burst = np.where(voltage_minima >= thresh)
    return indices_quiet, indices_burst

def calculate_ISI(M):
    '''calculates all ISIs lengths'''
    return np.diff(M.t)
    
def classify_ISI_lengths(M,Mv, thresh =-59.5):
    '''calculates ISI lengths and classifies them into quiet and burst'''
    indices_quiet, indices_burst = classify_ISI_indices(M,Mv, thresh)
    ISI = calculate_ISI(M)
    ISI_quiet = ISI[indices_quiet]
    ISI_burst = ISI[indices_burst]
    return ISI, ISI_quiet, ISI_burst
    
def plot_histograms(M,Mv, thresh =-59.5):
    '''plots histogram for ISIs and classified ISIs'''
    ISI, ISI_quiet, ISI_burst = classify_ISI_lengths(M, Mv, thresh)
        
    
    plt.subplot(1,3,1)
    plt.title('All '+ str(ISI.shape[0]) + ' ISIs. ')
    plt.hist(ISI)
    plt.subplot(1,3,2)
    plt.title(str(ISI_quiet.shape[0]) + ' Quiet ISIs')
    plt.hist(ISI_quiet)
    plt.subplot(1,3,3)
    plt.title(str(ISI_burst.shape[0]) + ' Burst ISIs')
    plt.hist(ISI_burst)
    

plot_everything()

        

    


