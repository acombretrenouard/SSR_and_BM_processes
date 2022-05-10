"""test the functions progressively"""


import matplotlib.pyplot as plt
import numpy as np
# adding my module to path
import sys
sys.path.append('/Users/antoine/Documents/X/3A/stages 3A/CSH Vienne/code')
import sample

## Progressive tests
"""this section enables to test various features of the code that were edited progressively, and to ensure that no conflits arise with previous work"""

def test00() :
    print("test 00 : building a 'System' object")
    syst = sample.core.System()
    print(syst.nbr)
    print(syst.state)
    print(syst.J_0)
    print(syst.noise)
    print(syst.run)
    syst.plotMatrix()
    return


def test01() :
    print('test 1 : instanciation of class System')
    syst = sample.core.System()
    a = syst.noise_inpt
    print(syst.noise_inpt)
    mat = sample.util.genNoise(nbr=4, rule='nrm')
    print(mat)
    return




def test02() :
    print('test 2 : System.doStep(), System.reset(), System.run(), syst.plotState()')
    plt.close('all')
    syst = sample.core.System()
    for i in range(5) :
        print('state at t=%i : '%i, syst.state)
        syst.doStep()
    syst.reset()
    print('.after reset : ', syst.state)
    syst.run()
    print('...after run : ', syst.state)
    print('state at t=4 : ', syst.states[:,4])
    syst.plotState()
    plt.show()
    return





def test03() :
    print('test 3 : Syste.animateState()')
    plt.close('all')
    syst = sample.core.System()
    syst.run()
    syst.animateState()
    plt.show()
    return




def test04() :
    print('test 4 : inputs')
    plt.close('all')
    syst = sample.core.System(nbr=30, dt=0.01, n_step=1000)
    syst.info()

    syst = sample.core.System(nbr=30, dt=0.25, end_time=100)
    syst.info()
    syst.plotMatrix()
    syst.run()
    syst.animateState(log=False)
    plt.show()
    return




def test05() :
    print("test 5 : printing runtime (in System.run(), set 'delay' to 0.01)")
    syst = sample.core.System(n_step=10000)
    syst.run()
    return





def test06() :
    print('test 6 : noise')
    plt.close('all')
    syst = sample.core.System(nbr=30, dt=0.01, end_time=100, dyn='mfd', noise='nrm', noise_inpt=(1.,10.))
    syst.run()
    syst.animateState()
    plt.show()
    return




def test07() :
    print('test 7 : BMtoolkit class, summing weights and plotting')
    plt.close('all')
    syst = sample.core.System(nbr=30, dt=0.025, end_time=99.976, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
    syst.info()
    syst.run()

    tlk = sample.core.BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.plotData(log=True, key='av_w')
    plt.show()
    return




def test08() :
    print('test 8 : normalized average weight')
    plt.close('all')
    syst = sample.core.System(nbr=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(1., 10.))
    syst.run()
    syst.info()

    tlk = sample.core.BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.plotData(log=True, key='av_w', ylabel='average wealth')
    tlk.normAvW()
    tlk.plotData(log=False, key='n_av_w')
    plt.show()
    return





def test09() :
    print('test 9 : rescale function')
    syst = sample.core.System(nbr=5, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
    syst.run()

    tlk = sample.core.BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    print(tlk.data['av_w'][:5])
    tlk.rescale()
    print(tlk.data['rsc_states'][:5])
    for i in [0,100,1000] :
        print(np.sum(tlk.data['rsc_states'][:,i]))
    return








def test10() :
    print('test 10 : testing genNoise() parameters')
    syst = System(nbr=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(1., 10.))
    syst.run()
    syst.info()
    for i in range(3) :
        mat = sample.util.genNoise(nbr=syst.nbr, rule=syst.noise, inpt=syst.noise_inpt)
        print(mat)
    return






def test11() :
    print('test 11 : time indexation')
    syst = sample.core.System(n_step=6, dt=0.3)
    syst.run()

    tf = syst.time
    end_time = syst.end_time
    T = syst.T
    Ts = np.linspace(0., end_time, T)
    syst.run()
    tf = syst.time
    print('Ts : ', Ts)
    print('tf : ', tf)
    print('end_time : ', end_time)
    print('states[0](t) : ', syst.states[0,:])
    return



def test12() :
    syst = sample.core.System(nbr=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(1., 10.))
    syst.run()

    T = syst.getTimes()
    Ys = np.copy(rescale(syst.states))

    # three agents' time series
    display(T, Ys[0], name='3 series', color='r')
    display(T, Ys[1], name='3 series', color='g')
    display(T, Ys[2], name='3 series', color='b', ylabel='rescaled weight')

    # all agents time serie
    displayAll(T, Ys, name='all series', ylabel='rescaled weight')

    # all time series, cumulated
    displayAll(T, cumul(Ys), name='all series cumul', ylabel='rescaled weight (cumulative)')

    # all wealth hisotgram, cumulated
    N = np.shape(Ys)[0]
    displayAll(np.arange(N), np.transpose(cumul(Ys)), name='all histograms cumul', ylabel='rescaled weight (cumulative)')

    # rescaled cov matrix
    displayMat(rcov(Ys), name='cov matrix')

    # autocorrelation
    display(T, rautocorr(Ys[0]), name='autocorr', color='r')
    display(T, rautocorr(Ys[1]), name='autocorr', color='g')
    display(T, rautocorr(Ys[2]), name='autocorr', color='b', ylabel='autocorrelation')

    # autocovariance
    display(T, rautocov(Ys[0]), name='autocov', color='r')
    display(T, rautocov(Ys[1]), name='autocov', color='g')
    display(T, rautocov(Ys[2]), name='autocov', color='b', ylabel='autocovariance')

    # variogram
    display(T, rvariogram(Ys[0]), name='variogram', color='r')
    display(T, rvariogram(Ys[1]), name='variogram', color='g')
    display(T, rvariogram(Ys[2]), name='variogram', color='b', ylabel='variogram')

    # Y_2 index
    display(T, Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # histogram
    displayHist(Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # other ?

    plt.show()
    return








def all() :
    test00()
    test01()
    test02()
    test03()
    test04()
    test05()
    test06()
    test07()
    test08()
    test09()
    test10()
    test11()
    test12()
    return


## Dev

## functions on a single time serie

def rautocorr(Y) :
    """Y is a 1-dimensionnal time sequence, returns the *rescaled* autocorrelation time-serie (of same length
! beware : the value for large lag are not trustable !"""
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
    """Y is a 1-dimensional time sequence, returns the *rescaled* autocovariance time-serie (of same length
! beware : the value for large lag are not trustable !"""
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
    """Y is a 1-dimensional time sequence, returns the *rescaled* variogram time-serie (of same length
! beware : the value for large lag are not trustable !"""
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

## function on a set of time series

from scipy.stats.mstats import gmean

def rcov(Ys) :
    """Ys is a N-dimensional time sequence, returns the *rescaled* covariance matrix
the time axis is axis 1"""
    cov = np.cov(Ys)
    gm = gmean(np.diagonal(cov))
    return cov/gm

def Y2(Ys) :
    """return the Y_2-index time serie as defined in BM2000"""
    Y2 = np.sum(Ys**2, axis=0)
    return Y2

def rescale(Ys) :
    """rescale each timestep by the average (agent wise)"""
    return Ys/np.mean(Ys, axis=0) # implicit np.multiply()

def cumul(Ys) :
    """returns the cumulative weights for 0 to N (along axis 0)"""
    return np.cumsum(Ys, axis=0)



## display functions

def display(T, Y, ylabel='', log=False, name='fig1', show=False, color='b') :
    """Y is a 1-dimensional time-serie, diplays it"""
    plt.figure(name)
    if log :
        plt.yscale('log')
    plt.plot(T, Y, color=color)
    plt.xlabel('time (s)')
    plt.ylabel(ylabel)
    if show : plt.show()
    return

def displayHist(Y, ylabel='', log=False, name='fig1', show=False) :
    """plots the histogram of the flattenned array Y"""
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


import matplotlib.cm # matplotlib colormaps

def displayAll(T, Ys, ylabel='', log=False, name='fig1', show=False) :
    """Ys is a N-dimensional time-serie, diplays all the curves in a bundle
the time axis is axis 1"""
    cmap = matplotlib.cm.get_cmap('plasma')
    N = np.shape(Ys)[0]
    for i in range(N) :
        rgb = cmap(i/N)[0:3]
        rgba = (rgb[0], rgb[1], rgb[2], 0.5)
        display(T, Ys[i], ylabel=ylabel, log=log, name=name, color=rgba)
    if show : plt.show()
    return

from mpl_toolkits.axes_grid1 import make_axes_locatable # for plt.colorbar axes positionning
import matplotlib.colors as colors

def displayMat(mat, name='fig1', show=False, style='cor') :
    """plots the matrix mat
style :
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


## test bench

syst = sample.core.System(nbr=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(1., 10.))
syst.run()

T = syst.getTimes()
Ys = np.copy(rescale(syst.states))



"""
# draft for two deltaHist() and plotDeltaHist() functions

syst = sample.core.System(nbr=30, dt=0.001, end_time=100, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.normAvW()

T = syst.T
n_Av_W = tlk.data['n_av_w']
deltas = n_Av_W[1:]-n_Av_W[:T-1]

plt.figure()
hist1 = plt.hist(n_Av_W, bins=100)

plt.figure()
hist2 = plt.hist(deltas, bins=100)



# draft for a toDrivingMatrix() function

syst = sample.core.System(nbr=30, dt=0.001, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = sample.core.BMtoolkit()
tlk.load(syst)

tlk.avW()
tlk.normAvW()

T = syst.T
n_Av_W = tlk.data['n_av_w']
deltas = n_Av_W[1:]-n_Av_W[:T-1]





# calculating the variogram and autocorrelations of n_Av_W(t) (the two are equivalent)

def diff(tab, tau) :
    L = np.shape(tab)[0]
    deltas = np.zeros(L)
    if tau < L :
        deltas[:L-tau] = tab[tau:]-tab[:L-tau]
    else :
        print('ERROR - diff() - index out of range, return np.zeros(%d)'%(L))
    return deltas

def mult(tab, tau) :
    L = np.shape(tab)[0]
    prod = np.zeros(L)
    if tau < L :
        prod[:L-tau] = tab[tau:]*tab[:L-tau]
    else :
        print('ERROR - mult() - index out of range, return np.zeros(%d)'%(L))
    return prod



syst = sample.core.System(nbr=30, dt=0.001, n_step=100000, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))

for i in range(3) :
    syst.reset()
    syst.run()
    tlk = sample.core.BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.normAvW()
    #tlk.plotNAvW()
    n_Av_W = tlk.data['n_av_w']
    # variogram
    avs = []
    trange = int(T/10)
    norm = np.average(n_Av_W[:T-trange]**2)
    for tau in range(trange) :
        deltas = diff(n_Av_W, tau)
        av = np.average(deltas[:T-trange]**2)/norm
        avs.append(av)
    taus = np.arange(trange)*syst.dt
    plt.figure('v')
    plt.plot(taus, avs)
    plt.xlabel('tau (s)')
    plt.ylabel('variogram')
    # correlogram
    avs = []
    norm = np.average(n_Av_W[:T-trange]**2)
    for tau in range(trange) :
        prod = mult(n_Av_W, tau)
        av = (np.average(prod[:T-trange]**2)-norm)/norm
        avs.append(av)
    plt.figure('c')
    plt.plot(taus, avs)
    plt.xlabel('tau (s)')
    plt.ylabel('correlogram')





# investigating the increments

syst = sample.core.System(nbr=30, dt=0.001, n_step=100000, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))

for i in range(3) :
    syst.reset()
    syst.run()
    tlk = sample.core.BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.normAvW()
    #tlk.plotNAvW()
    n_Av_W = tlk.data['n_av_w']
    increments = n_Av_W[1:]-n_Av_W[:syst.T-1] # weird
    # variogram
    avs = []
    trange = int(T/10)
    norm = np.average(increments[:T-trange]**2)
    for tau in range(trange) :
        deltas = diff(increments, tau)
        av = np.average(deltas[:syst.T-trange]**2)/norm
        avs.append(av)
    taus = np.arange(trange)*syst.dt
    plt.figure('v')
    plt.plot(taus, avs)
    plt.xlabel('tau (s)')
    plt.ylabel('variogram')
    # correlogram
    avs = []
    norm = np.average(increments[:syst.T-1-trange]**2)
    for tau in range(trange) :
        prod = mult(increments, tau)
        av = np.average(prod[:syst.T-1-trange]**2)/norm
        avs.append(av)
    plt.figure('c')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(taus, avs)
    plt.xlabel('tau (s)')
    plt.ylabel('correlogram')




# draft for studying max/min

syst = sample.core.System(nbr=300, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.ratio()
tlk.plotData(log=False, key='ratio')

ratios = tlk.data['ratio']

plt.figure()
plt.xscale('log')
plt.yscale('log')
l_bins = np.logspace(1,5,100)
hist1 = plt.hist(ratios, bins=l_bins)




# exp 1 : From when do we have a stable histogram for an agent ?
syst = sample.core.System(nbr=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()


#
%matplotlib notebook

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')


#
syst = sample.core.System(nbr=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
syst.run()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')






#
syst = sample.core.System(nbr=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()
# plotting histogram for ||J_0|| ~ 1 Hz

%matplotlib inline

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
# reference curves
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')
# histogram for agent n°0 for the first second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,:1000], bins=100, label='agent n°0 for the first second')
plt.legend()
plt.show()
# histogram for agent n°0 the sixth second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,50000:60000], bins=100, label='agent n°0 the sixth second')
plt.legend()
plt.show()
# histogram for agent n°0 the tenth second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,90000:100000], bins=100, label='agent n°0 the tenth second')
plt.legend()
plt.show()
# histogram for agent n°0 the last ten seconds
plt.figure()
plt.hist(tlk.data['rsc_states'][0,100000:200000], bins=100, label='agent n°0 the last ten seconds')
plt.legend()
plt.show()
# histogram for agent n°6 from t=3s to t=20s
plt.figure()
plt.hist(tlk.data['rsc_states'][6,30000:200000], bins=100, label='agent n°6 from t=3s to t=20s')
plt.legend()
plt.show()





# plotting a histogram for ||J_0|| ~ 10 Hz

syst.rebuildMatrix(p=10)
syst.reset()
syst.run()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
# reference curves
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')
# histogram for agent n°0 for the first second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,:1000], bins=100, label='agent n°0 for the first second')
plt.legend()
plt.show()
# histogram for agent n°0 the sixth second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,50000:60000], bins=100, label='agent n°0 the sixth second')
plt.legend()
plt.show()
# histogram for agent n°0 the tenth second
plt.figure()
plt.hist(tlk.data['rsc_states'][0,90000:100000], bins=100, label='agent n°0 the tenth second')
plt.legend()
plt.show()
# histogram for agent n°0 the last ten seconds
plt.figure()
plt.hist(tlk.data['rsc_states'][0,100000:200000], bins=100, label='agent n°0 the last ten seconds')
plt.legend()
plt.show()
# histogram for agent n°6 from t=3s to t=20s
plt.figure()
plt.hist(tlk.data['rsc_states'][6,30000:200000], bins=100, label='agent n°6 from t=3s to t=20s')
plt.legend()
plt.show()







# experiment 2 : plotting cov. matrix of states

syst = sample.core.System(nbr=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
syst.run()




#... continuing over last cell

%matplotlib inline

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()

T = syst.T
dt = syst.dt
end_time = syst.end_time


# reference curves
#tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
#tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
#tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
#tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')

# \/ test \/
def plotCov(toolkt, inf=0, sup=None) :
    if sup != None :
        mat = np.cov(toolkt.data['rsc_states'][0:5,inf:sup])
    else :
        mat = np.cov(toolkt.data['rsc_states'][0:5,inf:])
    # plotting
    plt.figure()
    ax = plt.subplot()
    norm = colors.CenteredNorm()
    im = ax.matshow(mat, cmap='bwr', norm=norm)
    # set colorbar (scale) next to matrix
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # show
    plt.show()

# exp 2 : cutting the relaxation part is necessary
plotCov(tlk, inf=0)
#plotCov(tlk, inf=100)
plotCov(tlk, inf=1000)
#plotCov(tlk, inf=10000)
plotCov(tlk, inf=100000)





# exp 2bis : for short timescales, less correlated !!! (non-intuitive) :
plotCov(tlk, inf=100000, sup=100010)
plotCov(tlk, inf=100000, sup=100100)
plotCov(tlk, inf=100000, sup=101000)
plotCov(tlk, inf=100000, sup=110000)
plotCov(tlk, inf=100000, sup=200000)








# exp 2ter : show that correlations are driven by the matrix

syst = sample.core.System(nbr=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
i1=0
i2=3
syst.J_0[i1,i2] += 100
syst.J_0[i2,i2] -= 100
syst.J_0[i2,i1] += 100
syst.J_0[i1,i1] -= 100
syst.run()







#...and we see the correlation

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()

plotCov(tlk, inf=100000)









# experiment 3 : noise must surely lower correlations TO CONTINUE


syst = sample.core.System(nbr=30, dt=0.01, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(1.,10.))
syst.rebuildMatrix(p=0.1)
syst.run()
syst.info()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.plotData(log=True, key='av_w')







syst = sample.core.System(nbr=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.1, 10.))
syst.run()
syst.info()

tlk = sample.core.BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.plotData(log=True, key='av_w')

"""
