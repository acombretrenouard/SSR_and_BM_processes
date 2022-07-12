"""core functions usefull for modelling the system or analysing results..."""

## Imports

# general modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # for plt.colorbar axes
import matplotlib.colors as colors
import matplotlib.animation as animation
import os
import time
import copy

# own modules
# \/ normal \/
#from . import util
# \/ for testing \/
import sys
sys.path.append('/Users/antoine/Documents/X/3A/stages_3A/CSH_Vienne/code')
import sample.util as util




## Body


class System(object):
    """sample.core.System(self, n_step=100, end_time=10)

Defines the parents class common to all systems (SSR, BM, largeBM...).
Baseline state is 1-dimensional.

See classes BM and largeBM for details."""

    def __init__(self, n_step=100, end_time=10) :
        # current step
        self.step = 0
        # max step and/or time
        self.n_step = n_step
        self.end_time = end_time
        # state and memory
        self.time = 0.
        self.times = []
        self.times.append(0.)
        self.state = 0.
        self.states = []
        self.states.append(0.)
        # misc
        self.running = False
        return

    def info(self) :
        """Prints info about the instance of class System."""
        return

    def getTime(self) :
        """Returns a copy of the current time."""
        return copy.copy(self.time)

    def getTimes(self) :
        """Returns a copy of all the sample times (independent instance)."""
        return copy.copy(self.times)

    def getState(self) :
        """Returns a copy of the curent state."""
        return copy.copy(self.state)

    def getStates(self) :
        """Returns a copy of all the sampled states (independent instance)."""
        return copy.copy(self.states)

    def doStep(self) :
        """Forwards the state by 0.1.
Before System.doStep() : System.time is the index of input variables (state(t), J(t)...)
After System.doStep() : System.time is the index of output variables (state(d+dt))."""
        # calc
        self.state = np.sin(self.time+0.1) # self.state(t+dt) = np.sin(t+dt)
        # increment
        self.time += 0.1
        self.step += 1
        # memory
        self.states.append(self.state)
        self.times.append(self.time)
        return

    def run(self) :
        """Calculates all states from n°1 to n°T-1 (n°0 is set by default).
Calls a thread loop that shows progress if long simulation."""
        start = time.time()
        # parallel thread
        delay = 1.
        timer = util.RepeatTimer(delay, util.progress, (self,))
        timer.start()
        # main thread
        self.running = True
        while self.step < self.n_step and self.time < self.end_time :
            self.doStep()
        self.running = False
        timer.cancel()
        delta = time.time() - start
        print('exit System.run(), runtime = ' + '{:.3f}'.format(delta) + ' s')
        return



    def live(self, log=False) :
        """Animates the simulation once done, depending on wich system it is."""
        return



class BM(System):
    """sample.core.BM(self, n_step=100, dt=0.1, end_time=-1., nbr=5, m=0.1, sigma=2., J=1.)


Description :
-----
Models the finite and not rescaled Bouchaud-Mézard model as described in Bouchaud-Mézard (2000) in Eq. 2. Uses the Ito-stochastic differential equation formalism.

The SDE that is modelled is :
    dW_i = A_i*dt + B_ii*dµ_i
where µ_i(t) are the Wiener processes.
We have :
    A_i = ∑(J_ij*W_j) - ∑(J_ji*W_i) + (m + sigma^2)*W_i = mat*W
    B_ii = sqrt(2)*sigma*W_i


Inputs :
-----
    int n_step     : number of step that will be simulated
    float dt       : time resolution
    float end_time : duration of the simulation (optionnal, has priority over the specified 'n_step')
    int nbr        : number of coordinates for a state
    float m        : 'm' paramater in Bouchaud, Mézard (2000)
    float sigma    : 'sigma' paramater in Bouchaud, Mézard (2000)
    """

    def __init__(self, n_step=100, dt=0.1, end_time=-1., nbr=5, m=0.1, sigma=2., J=1.) :
        System.__init__(self, n_step=n_step, end_time=end_time)
        # simulation parameters
        self.dt = dt
        if end_time > 0 :
            self.end_time = end_time
            self.n_step = int(end_time/dt)
        else :
            self.end_time = (self.n_step)*self.dt
        # state initialization
        self.nbr = nbr
        self.m = m
        self.sigma = sigma
        self.J = J
        self.state = np.zeros(self.nbr, dtype='float')
        self.state += 1.
        self.states = []
        self.states.append(copy.copy(self.state))
        # dynamics
        self.mat = self.J/self.nbr*np.ones((self.nbr, self.nbr)) + (-(self.nbr-1)*self.J/self.nbr + self.m + self.sigma**2)*np.eye(self.nbr)
        return

    def info(self) :
        """Prints info about the instance of class BM."""
        if len(str(self.state)) > 36 :
            st = str(self.state)[:36]+'...'
        else :
            st = str(self.state)
        print("\n Object 'BM' n°" + str(id(self)))
        print("-------------------------------------------------------------")
        print(' Simulation parameters\n     samples      : %d\n     length       : %f s\n     timestep     : %f s'%(self.n_step, self.end_time, self.dt))
        print(' System parameters\n     nbr : ' + str(self.nbr) + '\n     m   : ' + str(self.m) + '\n     sigma   : ' + str(self.sigma) + '\n     J   : ' + str(self.J))
        print(' Current state\n     current step : %d\n     current time : %f s\n     state        : '%(self.step, self.time) + st)
        print("-------------------------------------------------------------\n\n")
        return

    def doStep(self, rescale) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t). J(t)...)
After System.doStep() : System.time is the index of output variables (state(d+dt)).
No input, no output."""
        # gen noise (! the Wiener process has a standard deviation ('scale') of sqrt(dt) between t and t+dt !)
        dmu = np.random.normal(scale=np.sqrt(self.dt), size=self.nbr)
        # master equation : state(t+dt) = state(t) + dt.mat*W + sqrt(2)*sigma.(diag(W)*dmu)
        self.state += self.dt*np.dot(self.mat, self.state) + np.sqrt(2)*self.sigma*np.multiply(self.state, dmu)
        if rescale :
            self.state = self.state/np.average(self.state)
        # increment
        self.time += self.dt
        self.step += 1
        # memory
        self.states.append(self.getState())
        self.times.append(self.getTime())
        return

    def run(self, rescale=False) :
        """Calculates all states from n°1 to n°T-1 (n°0 is set by default).

Calls a thread loop that shows progress if long simulation.
No input, no output."""
        start = time.time()
        # parallel thread
        delay = 1.
        timer = util.RepeatTimer(delay, util.progress, (self,))
        timer.start()
        # main thread
        self.running = True
        #self.step = 1 # !! self.step représente le temps présent à l'entrée dans System.doStep() !!
        while self.step < self.n_step :
            self.doStep(rescale)
        self.running = False
        timer.cancel()
        delta = time.time() - start
        print('exit System.run(), runtime = ' + '{:.3f}'.format(delta) + ' s')
        return

    def live(self) :
        """Animates the simulation once done, depending on wich system it is."""
        return







class largeBM(System) :
    """sample.core.largeBM(self, n_step=100, dt=0.1, end_time=-1., nbr=5, noise='lBM', noise_inpt=(1.,10.))

Defines the uncorrelated version of the BM mode (see eq. 5 in BM2000). Uses the Ito-stochastic differential equation framework.

Input :
    int n_step     : number of step that will be simulated
    float dt       : time resolution
    float end_time : duration of the simulation (optionnal, has priority over the specified 'n_step')
    int nbr        : number of coordinates for a state
    float m        : 'm' paramater in Bouchaud, Mézard (2000)
    float sigma    : 'sigma' paramater in Bouchaud, Mézard (2000)"""

    def __init__(self, n_step=100, dt=0.1, end_time=-1., nbr=5, m=0.1, sigma=2., J=1.) :
        System.__init__(self, n_step=n_step, end_time=end_time)
        # simulation parameters
        self.dt = dt
        if end_time > 0 :
            self.end_time = end_time
            self.n_step = int(end_time/dt)
        else :
            self.end_time = (self.n_step)*self.dt
        # state initialization
        self.nbr = nbr
        self.m = m
        self.sigma = sigma
        self.J = J
        self.state = np.zeros(self.nbr, dtype='float')
        self.state += 1.
        self.states = []
        self.states.append(copy.copy(self.state))
        return


    def info(self) :
        """Prints info about the instance of class System."""
        if len(str(self.state)) > 36 :
            st = str(self.state)[:36]+'...'
        else :
            st = str(self.state)
        print("\n Object 'System' n°" + str(id(self)))
        print("-------------------------------------------------------------")
        print(' Simulation parameters\n     samples      : %d\n     length       : %f s\n     timestep     : %f s'%(self.n_step, self.end_time, self.dt))
        print(' System parameters\n     nbr : ' + str(self.nbr) + '\n     m   : ' + str(self.m) + '\n     sigma   : ' + str(self.sigma) + '\n     J   : ' + str(self.J))
        print(' Current state\n     current step : %d\n     current time : %f s\n     state        : '%(self.step, self.time) + st)
        print("-------------------------------------------------------------\n\n")
        return

    def doStep(self) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t), J(t)...).
After System.doStep()  : System.time is the index of output variables (state(d+dt)).
No input, no output."""
        # gen noise
        dmu = np.random.normal(scale=np.sqrt(self.dt), size=self.nbr) # ! the Wiener process has a standard deviation ('scale') of sqrt(dt) between t and t+dt !
        # master equation
        self.state += self.dt*self.J*(1.-self.state) + np.sqrt(2)*self.sigma*np.multiply(self.state, dmu) # driving equation : state(t+dt) = state(t) + dt.mat*W + sqrt(2)*sigma.(diag(W)*dmu)
        # increment
        self.time += self.dt
        self.step += 1
        # memory
        self.states.append(self.getState())
        self.times.append(self.getTime())
        return

    def run(self) :
        """Calculates all states from n°1 to n°T-1 (n°0 is set by default).

Calls a thread loop that shows progress if long simulation.
No input, no output."""
        start = time.time()
        # parallel thread
        delay = 1.
        timer = util.RepeatTimer(delay, util.progress, (self,))
        timer.start()
        # main thread
        self.running = True
        #self.step = 1 # !! self.step représente le temps présent à l'entrée dans System.doStep() !!
        while self.step < self.n_step :
            self.doStep()
        self.running = False
        timer.cancel()
        delta = time.time() - start
        print('exit System.run(), runtime = ' + '{:.3f}'.format(delta) + ' s')
        return

    def live(self) :
        """Animates the simulation once done, depending on wich system it is."""
        return





class SSR(System) :

    def __init__(self, nbr=10, rate=1., n_step=100, dt_type='cst', drive_type='top', choice_func=util.Choice.cst) :
        System.__init__(self)
        self.nbr = nbr
        self.rate = rate
        self.dt_type = dt_type
        self.drive_type = drive_type
        self.choice_func = choice_func
        self.n_step = n_step
        return

    def info(self) :
        return

    def doStep(self) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t). J(t)...)
After System.doStep() : System.time is the index of output variables (state(d+dt)).
No input, no output.


TO DO :
    - write a function choice(state) that returns TRUE if the driving should happen,
    - write a function drive(state, n_states) that picks the next state,
    - ..."""
        old_state = self.state
        # choice
        drive = self.choice_func(self.state, self.nbr)
        # driving
        if drive :
            if self.drive_type == 'unif' :
                self.state = np.random.randint(0, self.nbr)
            elif self.drive_type == 'top' :
                self.state = self.nbr-1
            elif self.drive_type == 'gnd' and self.state == 0 :
                self.state = self.nbr-1
            else :
                # no driving
                pass
        # dissipation
        else :
            if self.state > 0 :
                self.state = np.random.randint(0, self.state) #.. in [[0, self.state[[
            else :
                # we already have self.state == 0
                pass
        # waiting time
        if self.dt_type == 'exp' :
            dt = np.random.exponential(scale=self.rate)
        elif self.dt_type =='varying_exp' :
            dt = np.random.exponential(scale=self.rate/(old_state+1))
        else :
            # constant timestep
            dt = self.rate**-1
        # increment
        self.time += dt
        self.step += 1
        # memory
        self.states.append(self.getState())
        self.times.append(self.getTime())
        return


    def run(self) :
        """Calculates all states from n°1 to n°T-1 (n°0 is set by default).

Calls a thread loop that shows progress if long simulation.
No input, no output."""
        start = time.time()
        # parallel thread
        delay = 1.
        timer = util.RepeatTimer(delay, util.progress, (self,))
        timer.start()
        # main thread
        self.running = True
        #self.step = 1 # !! self.step représente le temps présent à l'entrée dans System.doStep() !!
        while self.step < self.n_step :
            self.doStep()
        self.running = False
        timer.cancel()
        delta = time.time() - start
        print('exit System.run(), runtime = ' + '{:.3f}'.format(delta) + ' s')
        return





## Testing






def test01() :
    """test 1 : parent class System"""
    syst = System()
    syst.run()
    plt.plot(syst.times, syst.states)
    plt.title('Baseline simulation for System()')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()
    print('TEST 01 OK\n---------------------------------\n\n\n')
    return




def test02() :
    """test 2 : input and run for BM"""
    syst = BM(nbr=30, dt=0.01, n_step=1000)
    syst.run()
    syst.info()
    syst = BM(nbr=30, dt=0.01, end_time=10)
    syst.info()
    syst.run()
    syst.info()
    st = syst.getState()
    plt.plot(np.arange(30), st)
    plt.show()
    print('TEST 02 OK\n---------------------------------\n\n\n')
    return


def test03() :
    """test 3 : extensive plotting"""
    syst = BM(nbr=4, dt=0.01, end_time=10)
    syst.run()

    T = syst.getTimes()
    Ys = syst.getStates()
    Ys = np.transpose(np.array(Ys))
    Ys = util.rescale(Ys)

    # three agents' time series
    util.display(T, Ys[0], name='3 series', color='r')
    util.display(T, Ys[1], name='3 series', color='g')
    util.display(T, Ys[2], name='3 series', color='b', ylabel='rescaled weight')

    # all agents time serie
    util.displayAll(T, Ys, name='all series', ylabel='rescaled weight')

    # all time series, cumulated
    util.displayAll(T, util.cumul(Ys), name='all series cumul', ylabel='rescaled weight (cumulative)')

    # all wealth hisotgram, cumulated
    N = np.shape(Ys)[0]
    util.displayAll(np.arange(N), np.transpose(util.cumul(Ys)), name='all histograms cumul', ylabel='rescaled weight (cumulative)')

    # rescaled cov matrix
    util.displayMat(util.rcov(Ys), name='cov matrix')

    # autocorrelation
    util.display(T, util.rautocorr(Ys[0]), name='autocorr', color='r')
    util.display(T, util.rautocorr(Ys[1]), name='autocorr', color='g')
    util.display(T, util.rautocorr(Ys[2]), name='autocorr', color='b', ylabel='autocorrelation')

    # autocovariance
    util.display(T, util.rautocov(Ys[0]), name='autocov', color='r')
    util.display(T, util.rautocov(Ys[1]), name='autocov', color='g')
    util.display(T, util.rautocov(Ys[2]), name='autocov', color='b', ylabel='autocovariance')

    # variogram
    util.display(T, util.rvariogram(Ys[0]), name='variogram', color='r')
    util.display(T, util.rvariogram(Ys[1]), name='variogram', color='g')
    util.display(T, util.rvariogram(Ys[2]), name='variogram', color='b', ylabel='variogram')

    # Y_2 index
    util.display(T, util.Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # histogram
    util.displayHist(util.Y2(Ys), name='Y2_hist', ylabel='$Y_2$ index')

    plt.show()
    print('TEST 03 OK\n---------------------------------\n\n\n')
    return







def test04() :
    """test 4 : testing largeBM instanciation"""
    syst = largeBM(nbr=30, dt=0.01, n_step=1000)
    syst.run()
    syst.info()
    syst = largeBM(nbr=30, dt=0.01, end_time=10)
    syst.info()
    syst.run()
    syst.info()
    st = syst.getState()
    plt.plot(np.arange(30), st)
    plt.show()
    print('TEST 04 OK\n---------------------------------\n\n\n')
    return



def test05() :
    """test 5 : testing largeBM"""
    syst = largeBM(nbr=4, dt=0.01, end_time=20, noise_inpt=(1., 1.))
    syst.run()

    T = syst.getTimes()
    Ys = syst.getStates()
    Ys = np.transpose(np.array(Ys))

    # three agents' time series
    util.display(T, Ys[0], name='3 series', color='r')
    util.display(T, Ys[1], name='3 series', color='g')
    util.display(T, Ys[2], name='3 series', color='b', ylabel='weight')

    # all agents time serie
    util.displayAll(T, Ys, name='all series', ylabel='weight')

    # all time series, cumulated
    util.displayAll(T, util.cumul(Ys), name='all series cumul', ylabel='weight (cumulative)')

    # all wealth hisotgram, cumulated
    N = np.shape(Ys)[0]
    util.displayAll(np.arange(N), np.transpose(util.cumul(Ys)), name='all histograms cumul', ylabel='weight (cumulative)')

    # rescaled cov matrix
    util.displayMat(util.rcov(Ys), name='cov matrix')

    # autocorrelation
    util.display(T, util.rautocorr(Ys[0]), name='autocorr', color='r')
    util.display(T, util.rautocorr(Ys[1]), name='autocorr', color='g')
    util.display(T, util.rautocorr(Ys[2]), name='autocorr', color='b', ylabel='autocorrelation')

    # autocovariance
    util.display(T, util.rautocov(Ys[0]), name='autocov', color='r')
    util.display(T, util.rautocov(Ys[1]), name='autocov', color='g')
    util.display(T, util.rautocov(Ys[2]), name='autocov', color='b', ylabel='autocovariance')

    # variogram
    util.display(T, util.rvariogram(Ys[0]), name='variogram', color='r')
    util.display(T, util.rvariogram(Ys[1]), name='variogram', color='g')
    util.display(T, util.rvariogram(Ys[2]), name='variogram', color='b', ylabel='variogram')

    # Y_2 index
    util.display(T, util.Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # histogram
    util.displayHist(util.Y2(Ys), name='Y2_hist', ylabel='$Y_2$ index')

    plt.show()
    print('TEST 05 OK\n---------------------------------\n\n\n')
    return



def test06() :
    """test 6 : testing SSR instanciation"""
    syst = SSR(nbr=30, n_step=1000)
    syst.run()
    syst.info()
    syst = SSR(nbr=300, n_step=100, dt_type='cst', drive_type='unif')
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.step(ts, st)
    plt.show()
    print('TEST 06 OK\n---------------------------------\n\n\n')
    return


def test07() :
    """test 7 : testing SSR histograms"""
    # cst-unif
    syst = SSR(nbr=300, n_step=1000, dt_type='cst', drive_type='unif')
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.figure('cst-unif')
    util.plotHistogram(ts,st, log=True)
    # exp-unif
    syst = SSR(nbr=300, n_step=1000, dt_type='exp', drive_type='unif')
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.figure('exp-unif')
    util.plotHistogram(ts,st, log=True)
    # varying_exp-unif
    syst = SSR(nbr=300, n_step=1000, dt_type='varying_exp', drive_type='unif')
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.figure('varying_exp-unif')
    util.plotHistogram(ts,st, log=True)
    # cst-top
    syst = SSR(nbr=300, n_step=1000, dt_type='cst', drive_type='top')
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.figure('cst-top')
    util.plotHistogram(ts,st, log=True)
    # show
    #plt.show()
    print('TEST 07OK\n---------------------------------\n\n\n')
    return

# test 8: ...
def test08() :
    # lambda = 0.9
    lmda = 0.9
    syst = SSR(nbr=300, n_step=10000, dt_type='cst', drive_type='unif', choice_func=util.Choice.buildCst(lmda))
    syst.run()
    ts1 = syst.getTimes()
    st1 = syst.getStates()
    # lambda = 0.5
    lmda = 0.5
    syst = SSR(nbr=300, n_step=10000, dt_type='cst', drive_type='unif', choice_func=util.Choice.buildCst(lmda))
    syst.run()
    ts2 = syst.getTimes()
    st2 = syst.getStates()
    # time serie
    plt.figure('time series')
    plt.step(ts1[:100], st1[:100], label = '$\lambda = 0.9$')
    plt.step(ts2[:100], st2[:100], label = '$\lambda = 0.5$')
    plt.xlabel('step n°')
    plt.ylabel('state n°')
    plt.legend()
    # histogram
    plt.figure('lambda = 0.9')
    plt.grid()
    util.plotHistogram(ts1,st1, log=True)
    plt.figure('lambda = 0.5')
    plt.grid()
    util.plotHistogram(ts2,st2, log=True)
    # show
    plt.show()
    print('TEST 08 OK\n---------------------------------\n\n\n')
    return




# test NNNNN: ...
def test0NNNNN() :
    #...
    print('TEST 0NNNNN OK\n---------------------------------\n\n\n')
    return



## Dev
"""edit here the code that should be added to this file"""