"""utility functions : build, in/out, etc..."""

## Imports

import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import matplotlib.cm # matplotlib colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable # for plt.colorbar axes positionning
import matplotlib.colors as colors


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
        factor = (np.ones((nbr, nbr)) - np.eye(nbr))*0.99 + np.eye(nbr)
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
the time axis is axis 1."""
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
