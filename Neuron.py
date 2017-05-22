from brian2 import *
import numpy as np


plt.rcParams['figure.figsize'] = 12, 4

defaultclock.dt = 0.05*ms

Cm = 1.65*uF # /cm**2
Iapp = .158*uA
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens

tau=1*ms
I_noise = .06*uA

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

neuron = NeuronGroup(1, eqs,  threshold = 'v >-.01*volt', refractory = 'v > -.01*volt')

#neuron.v = [-62,-58.5,-55,-60,-53.5,-52]*mV 
#neuron.n = [-.1, -.1, -.1, .5, .5, .5]

neuron.v = -55*mV 
neuron.n = .1

M = SpikeMonitor(neuron, variables = 'v')
Mv = StateMonitor(neuron, 'v', record=True)
Mn = StateMonitor(neuron, 'n', record=True)

run(500000*ms, report='text')

plt.figure()
plt.title(str(Cm) + ' ' + str(Iapp)+' '+ str(I_noise))
plt.plot(Mv.t/ms, Mv.v.T/mV)
plt.show()

plt.subplot(1,2,1)
plt.plot(Mv.v.T/mV,Mn.n.T)
plt.subplot(1,2,2)
plt.plot(Mv.v.T/mV,Mn.n.T)
plt.xlabel('mV')
plt.ylabel('n')
plt.xlim((-70,-50))
plt.ylim((-0.1,1))
plt.show()


def classify_ISI(M,Mv, thresh):
    '''returns tuple: indices of spikes after which there is a quiet interval,
                      indices of spikes after which there is a burst interval '''
    below_thresh=np.where((Mv.v[0]/mV < thresh) & (Mv.t <= M.t[-1]) & (Mv.t >= M.t[0]))
    indices_quiet = np.unique(np.searchsorted(M.t/ms,np.array(Mv.t/ms)[below_thresh]))-1
    indices_burst = np.delete(np.arange(len(M.t/ms)-1), indices_quiet)
    return indices_quiet, indices_burst
    
def plot_histograms(M,Mv, thresh =-59.5):
    indices_quiet, indices_burst = classify_ISI(M,Mv, thresh)
    ISI = np.diff(M.t/ms)
    ISI_quiet = ISI[indices_quiet]
    ISI_burst = ISI[indices_burst]
       
    plt.subplot(1,3,1)
    plt.title('All '+ str(ISI.shape[0]) + ' ISIs. ')
    plt.hist(ISI)
    plt.subplot(1,3,2)
    plt.title(str(ISI_quiet.shape[0]) + ' Quiet ISIs')
    plt.hist(ISI_quiet)
    plt.xlabel('Minimal ISI is ' + str(np.min(ISI_quiet)))
    plt.subplot(1,3,3)
    plt.title(str(ISI_burst.shape[0]) + ' Burst ISIs')
    plt.hist(ISI_burst)

plot_histograms(M,Mv) 

