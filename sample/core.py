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
sys.path.append('/Users/antoine/Documents/X/3A/stages 3A/CSH Vienne/code')
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
    """sample.core.BM(self, n_step=100, dt=0.1, end_time=-1., nbr=5, dyn='mfd', dyn_inpt=1., noise='BMs', noise_inpt=(1.,10.))

Defines a dynamical system with a macrostate distribution function and a first-order linear master equation. Intended to model the Bouchaud-Mézard model.

Inputs :
    int n_step     : number of step that will be simulated
    float dt       : time resolution
    float end_time : duration of the simulation (optionnal, has priority over the specified 'n_step')
    int nbr        : number of coordinates for a state
    str dyn        : type of dynamic used for the master equation
    int dyn_inpt   : parameter for the dynamical rule (matrix). Default is 1.
    str noise      : type of noise, '' if no noise
    tuple noise_inpt : parameters for the noise generating function
    """

    def __init__(self, n_step=100, dt=0.1, end_time=-1., nbr=5, dyn='mfd', dyn_inpt=1., noise='BMs', noise_inpt=(1.,10.)) :
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
        self.state = np.zeros(self.nbr, dtype='float')
        self.state += 1.
        self.states = []
        self.states.append(copy.copy(self.state))
        # dynamics
        self.dyn = dyn
        self.noise = noise
        self.noise_inpt = noise_inpt
        self.J_0 = util.buildMatrix(nbr=self.nbr, dyn=self.dyn, param=dyn_inpt)
        self.eta = np.zeros((self.nbr, self.nbr))
        self.etas = np.zeros((self.nbr, self.nbr, self.n_step)) # storage
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
        print(' System parameters\n     dynamic type : ' + self.dyn + '\n     noise type   : ' + self.noise + '\n     noise input  : ' + str(self.noise_inpt))
        print(' Current state\n     current step : %d\n     current time : %f s\n     state        : '%(self.step, self.time) + st)
        print("-------------------------------------------------------------\n\n")
        return

    def doStep(self) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t). J(t)...)
After System.doStep() : System.time is the index of output variables (state(d+dt)).
No input, no output."""
        # setting the matrix
        if self.noise != 'no noise' :
            self.eta = util.genNoise(nbr=self.nbr, rule=self.noise, inpt=self.noise_inpt)
            self.etas[:,:,self.step] = self.eta # storage
            J = self.J_0 + self.eta
        else :
            J = self.J_0
        # master equation
        self.state += self.dt*np.dot(J, self.state) # driving equation : state(t+dt) = state(t) + dt*J(t)*state(t)
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







class largeBM(System) :
    """sample.core.largeBM(self, n_step=100, dt=0.1, end_time=-1., nbr=5, noise='lBM', noise_inpt=(1.,10.))

Defines the uncorrelated version of the BM mode (see eq. 5 in BM2000).

Input :
    int n_step     : number of step that will be simulated
    float dt       : time resolution
    float end_time : duration of the simulation (optionnal, has priority over the specified 'n_step')
    int nbr        : number of coordinates for a state
    str noise      : type of noise, '' if no noise
    float dyn_inpt : parameter for the markovian matrix
    tuple noise_inpt : parameters for the noise generating function"""

    def __init__(self, n_step=100, dt=0.1, end_time=-1., nbr=5, dyn_inpt=1., noise='lBM', noise_inpt=(1.,10.)) :
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
        self.state = np.zeros(self.nbr, dtype='float')
        self.state += 1.
        self.states = []
        self.states.append(copy.copy(self.state))
        # dynamics
        self.J = dyn_inpt
        self.noise = noise
        self.noise_inpt = noise_inpt
        self.eta = np.zeros(self.nbr)
        self.etas = np.zeros((self.nbr, self.n_step)) # storage
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
        print(' System parameters\n     noise type   : ' + self.noise + '\n     noise input  : ' + str(self.noise_inpt))
        print(' Current state\n     current step : %d\n     current time : %f s\n     state        : '%(self.step, self.time) + st)
        print("-------------------------------------------------------------\n\n")
        return

    def doStep(self) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t), J(t)...).
After System.doStep()  : System.time is the index of output variables (state(d+dt)).
No input, no output."""
        # setting the noise vector
        self.eta = util.genNoise(nbr=self.nbr, rule=self.noise, inpt=self.noise_inpt, retmat=False)
        self.etas[:,self.step] = self.eta # storage
        # master equation
        self.state += self.dt*(np.multiply(self.state, self.eta) + self.J*(1-self.state)) # driving equation : state[i](t+dt) = state[i](t) + dt*(state[i](t)*noise + J(1-state[i](t))) with noise = eta[i](t) - m -sigm**2
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

    def __init__(self, nbr=10, rate=1., n_step=100) :
        System.__init__(self)
        self.nbr = nbr
        self.rate = rate
        return

    def info(self) :
        return

    def doStep(self) :
        """Forwards the state by dt.

Before System.doStep() : System.time is the index of input variables (state(t). J(t)...)
After System.doStep() : System.time is the index of output variables (state(d+dt)).
No input, no output."""
        # update state
        if self.state == 0 :
            self.state = self.nbr - 1
            dt = np.random.exponential(scale=self.rate*1)
        else :
            dt = np.random.exponential(scale=self.rate/self.state)
            self.state = np.random.randint(0, self.state)
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
    syst = SSR(nbr=300, n_step=100)
    syst.run()
    ts = syst.getTimes()
    st = syst.getStates()
    plt.plot(ts, st)
    plt.show()
    print('TEST 06 OK\n---------------------------------\n\n\n')
    return

# test NNNNN: ...
def test0NNNNN() :
    #...
    print('TEST 0NNNNN OK\n---------------------------------\n\n\n')
    return






## Dev
"""edit here the code that should be added to this file"""