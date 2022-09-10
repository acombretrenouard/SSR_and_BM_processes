"""utility functions : build, in/out, etc..."""

## Imports

import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import matplotlib.cm # matplotlib colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable # for plt.colorbar axes positionning
import matplotlib.colors as colors
import math


## BM utilities


def buildMatrix(nbr=5, dyn='uni', param=1.) :
    """sample.util.buildMatrix(nbr=5, dyn='uni', param=1.)

Builds a stochastic matrix, corresponding to the specified dynamic (with kwarg. 'dyn').

Input :
    int nbr : size of the matrix (square)
    str dyn : type of dynamics embodied by the matrix
    float param : parameter relevent to the dynamic's type (ex : for Bouchaud-Mézard model, this could be J).

Output :
    ndarray of shape (nbr, nbr)

Dynamical rules :
        uni --> uniformly in [0, 1[
        mfd --> mean field (all coef. equal to param/n_states, "param" is equivalent to "J")
        bin --> binomial (1 sample with probability param)
        pow --> power-tail law (Pareto here)
        nrm --> normal law
        ssr --> ssr process (with state no <-> energy)
        BGe --> Boltzmann-Gibbs equilibrium
        aph --> asymetrical mean-field model with an alpha parameter"""
    matrix = np.zeros((nbr,nbr), dtype=float)
    for j in range(nbr) :
        s = 0
        for i in range(nbr) :
            rate = 0.
            if   dyn=='mfd' : rate = param/nbr
            elif dyn=='uni' : rate = np.random.uniform()
            elif dyn=='bin' : rate = np.random.binomial(1,param)
            elif dyn=='pow' : rate = np.random.pareto(param)
            elif dyn=='nrm' : rate = np.abs(np.random.normal()) # ! the coefficients must be postive !
            elif dyn=='ssr' :
                if i<j : rate = 1/j
                elif (i==nbr-1 and j==0) : rate=1
            elif dyn=='BGe' : print('ERROR buildMatrix : dyn BGe not done yet')
            elif dyn=='aph' : rate = param/nbr
            else :
                print('DynamicalRule.initialize() ERROR : unknown distribution, uniform used instead\n Reminder :\n    uni --> uniformly in [0, 1[\n    bin --> binomial (1 sample)\n    pow --> power-tail law (Pareto here)\n    nrm --> normal law')
                rate = np.random.uniform()
            matrix[i,j] = rate
            if i != j : s += rate
        matrix[j,j] = - s # s is the sum of all other transition rates : the matrix is thus stochastic
    if dyn=='aph' :
        factor = (np.ones((nbr, nbr)) - np.eye(nbr))*0.9 + np.eye(nbr)
        matrix = np.multiply(matrix, factor)
    return matrix

def genNoise(nbr=5, rule='BMs', inpt=(1.,10.), retmat=True) :
    """sample.util.genNoise(nbr=5, rule='BMs', inpt=(1.,10.), retmat=True)

Generates a random matrix of shape (nbr, nbr) according to a certain rule ('rule').

Input :
    int nbr : size of vector / matrix generated
    str rule : specifies the noise generating rule according to the correspondance below
    tuple inpt : float parameters relevant to the distribution specified by 'rule'
    bool retmat : returns a matrix of size nbr by nbr if set to True. Returns a nbr-dimensional vector otherwise.

Output :
    ndarray of shape (nbr, nbr) or shape (nbr,) depending on 'retmat'.

Meaning of 'rule' :
    uni --> uniformly in [0, 1[
    bin --> binomial (1 sample)
    pow --> power-tail law (Pareto here)
    nrm --> normal law. INPUT : (m, sigm) := inpt
    BMs --> normal law. INPUT : as 'nrm' but scale = np.sqrt(2)*sigm
    lBM --> normal law. INPUT : as 'nrm' but scale = np.sqrt(2)*sigm and mean = -sigm**2"""
    if rule=='nrm' :
        mean, sigm = inpt
        diag = np.random.normal(loc=mean, scale=sigm, size=nbr)
    elif rule=='BMs' :
        mean, sigm = inpt
        diag = np.random.normal(loc=mean, scale=np.sqrt(2)*sigm, size=nbr)
    elif rule=='lBM' :
        mean, sigm = inpt
        diag = np.random.normal(loc=-sigm**2, scale=np.sqrt(2)*sigm, size=nbr)
    elif rule=='uni' :
        diag = np.random.uniform(size=nbr)
    elif rule=='bin' :
        diag = np.random.binomial(n=1, size=nbr)
    elif rule=='pow' :
        diag = np.random.pareto(size=nbr)
    else :
        print('ERROR - genNoise() - unknown noise rule, normal law used instead (mean=1., scale=10.)\nReminder :\n    uni --> uniformly in [0, 1[\n    bin --> binomial (1 sample)\n    pow --> power-tail law (Pareto here)\n    nrm --> normal law (scale=inpt[1]\n    BMs --> normal law where scale=np.sqrt(2)*inpt[1]')
        mean, sigm = 1.,10.
        diag = np.random.normal(loc=mean, scale=sigm, size=nbr)
    if retmat :
        return np.diag(diag)
    else :
        return diag

def progress(syst) :
    """Fancy printing of progress in a run of a simulation."""
    if syst.running :
        pc = 100.*float(syst.time)/float(syst.end_time)
        print('Run in progess : ' + "{:.1f}".format(pc) + ' %')
    else :
        print('Not running')
    return

class RepeatTimer(threading.Timer):
    """Parallel thread used to display runtime."""
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
        return




## Functions for 1D time serie

def rautocorr(Y) :
    """Y is a 1-dimensionnal time sequence, returns the *rescaled* autocorrelation time-serie (of same length).

! beware : the value for large lag-time are not trustable !"""
    list = []
    T = np.shape(Y)[0]
    for tau in range(T) :
        S = 0
        for t in range(T-tau) :
            S += Y[t+tau]*Y[t]
        S /= (T-tau)
        list.append(S)
    acr = np.array(list)
    var = np.var(Y)
    return  acr/var # implicit

def rautocov(Y) :
    """Y is a 1-dimensional time sequence, returns the *rescaled* autocovariance time-serie (of same length).

! beware : the value for large lag-time are not trustable !"""
    list = []
    T = np.shape(Y)[0]
    for tau in range(T) :
        S = 0
        for t in range(T-tau) :
            S += Y[t+tau]*Y[t]
        S /= (T-tau)
        list.append(S)
    acr = np.array(list)
    m = np.mean(Y)
    var = np.var(Y)
    return  (acr-m**2)/var # implicit

def rvariogram(Y) :
    """Y is a 1-dimensional time sequence, returns the *rescaled* variogram time-serie (of same length).

! beware : the value for large lag-time are not trustable !"""
    list = []
    T = np.shape(Y)[0]
    for tau in range(T) :
        S = 0
        for t in range(T-tau) :
            S += (Y[t+tau]-Y[t])**2
        S /= (T-tau)
        list.append(S)
    vario = np.array(list)
    var = np.var(Y)
    return  vario/var/2 # implicit

def doHist(Y, log=False) :
    """returns the histogram of the time serie (eventually flattened)"""
    if log : barg = np.logspace(np.log10(np.amin(copy)), np.log10(np.amax(copy)), num=100)
    hist, edges = np.histogram(Y, bins=barg, density=True)
    return hist, edges

## Function for multi-D time series

def rcov(Ys) :
    """Ys is a N-dimensional time sequence, returns the *rescaled* covariance matrix
the time axis is axis 1.

Rq : the rescale coefficient must be the Pearson correlation coefficient !!"""
    cov = np.cov(Ys)
    gm = gmean(np.diagonal(cov))
    return cov/gm

def Y2(Ys) :
    """Returns the Y_2-index time serie as defined in BM2000."""
    Y2 = np.sum(Ys**2, axis=0)
    return Y2

def rescale(Ys) :
    """Rescales each timestep by the average (agent wise).

Ex : if agents are indexed along axis 0 and timestamps along axis 1, this rescale each agent's weight by the average agent-wise weight."""
    return Ys/np.mean(Ys, axis=0) # implicit np.multiply()

def cumul(Ys) :
    """Returns the cumulative weights along axis 0 (summing from top to bottom)."""
    return np.cumsum(Ys, axis=0)



## Display functions

def display(T, Y, ylabel='', log=False, name='fig1', show=False, color='b') :
    """sample.util.display(T, Y, ylabel='', log=False, name='fig1', show=False, color='b')

Y is a 1-dimensional time-serie, diplays it."""
    plt.figure(name)
    if log :
        plt.yscale('log')
    plt.plot(T, Y, color=color)
    plt.xlabel('time (s)')
    plt.ylabel(ylabel)
    if show : plt.show()
    return

def displayHist(Y, ylabel='', log=False, name='fig1', show=False) :
    """sample.util.displayHist(Y, ylabel='', log=False, name='fig1', show=False)

Plots the histogram of the flattened array Y."""
    copy = Y.flatten()
    plt.figure(name)
    if log :
        plt.xscale('log')
        plt.yscale('log')
        barg = np.logspace(np.log10(np.amin(copy)), np.log10(np.amax(copy)), num=100)
    else :
        barg = 100
    plt.ylabel('density')
    hist = plt.hist(copy, bins=barg, density=True)
    del copy
    if show : plt.show()
    return

def displayAll(T, Ys, ylabel='', log=False, name='fig1', show=False) :
    """sample.util.displayAll(T, Ys, ylabel='', log=False, name='fig1', show=False)

Ys is a N-dimensional time-serie, diplays all the curves in a bundle.

The time axis is axis 1."""
    cmap = matplotlib.cm.get_cmap('plasma')
    N = np.shape(Ys)[0]
    for i in range(N) :
        rgb = cmap(i/N)[0:3]
        rgba = (rgb[0], rgb[1], rgb[2], 0.5)
        display(T, Ys[i], ylabel=ylabel, log=log, name=name, color=rgba)
    if show : plt.show()
    return


def displayMat(mat, name='fig1', show=False, style='cor') :
    """sample.util.displayMat(mat, name='fig1', show=False, style='cor')

Plots the matrix mat.

Rule for 'style' :
    'cor' for a cmap centered on 0
    ..."""
    # show matrix
    plt.figure(name)
    ax = plt.subplot()
    if style == 'cor' :
        norm = colors.CenteredNorm()
        im = ax.matshow(mat, cmap='bwr', norm=norm)
    else :
        im = ax.matshow(mat)
    # set colorbar (scale) next to matrix
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # show
    if show : plt.show()
    return






## Dev
"""edit here the code that should be added to this file"""



def plotHistogram(ts, st, log=False, add1=True) :
    """plot the histogram of occupancy in each state
log=True for log-scale

Input :
        ndarray ts : 1D list of timestamps (float)
        ndarray sts : 1D list of discrete states (int)    Rq : system jumps to state sts[i] at time ts[i]"""
    ts, st = np.array(ts), np.array(st)[:-1]
    N = np.shape(st)[0]
    if add1 : st+= 1
    deltas = ts[1:]-ts[:-1]
    if log :
        plt.xscale('log')
        plt.yscale('log')
        barg = np.logspace(np.log10(np.amin(st)), np.log10(np.amax(st)), num=50)
    else :
        barg = 50
    plt.hist(st, weights=deltas, density=True, log=log, bins=barg)
    plt.xlabel('state')
    plt.ylabel('relative time spent')
    return




def build_ER(nbr, p) :
    """build_ER(nbr, p)

returns an Erdos-Renyi network of size 'nbr' and of parameter 'p'

Return : symetric matrix of size nbr*nbr where a node is 1 with probability p (and 0 else)"""
    mat = np.zeros((nbr, nbr))
    for i in range(nbr) :
        for j in range(i+1, nbr) :
            choice = np.random.uniform() < p
            mat[i,j] = int(choice)
            mat[j,i] = int(choice)
    return mat


def treshold(Y) :
    """provides a treshold (lower bound) for estimating the tail exponent in BM simulations"""
    mn = np.mean(Y)
    max = np.amax(Y)
    # opt 1 (large value)
    return np.sqrt(max*mn)
    # opt 2 (> moyenne)
    #return mn

def estimate_exp(Y, tsld) :
    """estimate the tail-exponent for a stationnary process distributed broadly
ignores values under 'tsld' (treshold)"""
    rho = 1/(np.amax(Y) - tsld)
    counter = 0
    sum = 0
    for y in Y :
        if y >= tsld :
            counter += 1
            sum += np.log(y/tsld)
    lmda = 1 + counter/sum
    var = (lmda-1)**2/counter
    rho *= counter
    return lmda, var, rho

def ref_pow(lmda, tsld, rho) :
    func = lambda y : rho*(y/tsld)**-lmda
    return func


def ref_distr(mu) :
    Z = (mu-1)/math.gamma(mu)
    func = lambda w : Z*np.exp(-(mu-1)/w)/(w**(1+mu))
    return func


class Choice(object) :
    """collection of functions lambda(state) that will serve as choice functions for driving in sample.SSR.doStep()"""

    def cst(state, ref) :
        """returns True with propability 1-lmda regardless of the state"""
        lmda = 0.9
        return np.random.uniform() < 1-lmda

    def cst4(state, ref) :
        """returns True with propability 1-lmda regardless of the state"""
        lmda = 1-1/4
        return np.random.uniform() < 1-lmda

    def cst4XX(state, ref) :
        """returns True with propability 1-lmda regardless of the state"""
        lmda = 1-3/4000
        return np.random.uniform() < 1-lmda

    def buildCst(lmda) :
        """returns a function Choice.cst of parameter lmda"""
        func = lambda state, ref : np.random.uniform() < 1-lmda
        return func

    def upLin(state, ref) :
        """returns True with a linear probability"""
        lmda = 0.5*state/ref
        return np.random.uniform() < 1-lmda

    def downLin(state, ref) :
        """returns True with a linear probability"""
        lmda = 0.1*(1-state/ref)
        return np.random.uniform() < 1-lmda

    def gaussianFit(state, ref) :
        """returns True with propability b*(state/ref)**2"""
        b = 1
        lmda = b*(state/ref)**2
        return np.random.uniform() < 1-lmda

    def expFit(state, ref) :
        """returns True with propability b*(state/ref)**2"""
        b = 1
        lmda = b*state/ref
        return np.random.uniform() < 1-lmda

    def gammaFit(state, ref) :
        """returns True with propability b*(state/ref) - a + 1
a = 0.5
b = 0.5"""
        a = 0.9
        b = 0.1
        lmda = b*(state/ref) - a + 1
        return np.random.uniform() > lmda

    def invGammaFit(state, ref) :
        """returns True with propability b*(state/ref)**-1 - a + 1
a = 0.5
b = 0.5"""
        a = 0.9
        b = 0.1
        lmda = b*(ref/state) - a + 1
        return np.random.uniform() > lmda

    def bottom(state, ref) :
        if state==0 :
            return True
        else :
            return False
