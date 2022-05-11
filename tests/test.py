"""test the functions progressively"""


import matplotlib.pyplot as plt
import numpy as np
# adding my module to path
import sys
sys.path.append('/Users/antoine/Documents/X/3A/stages 3A/CSH Vienne/code')
import sample

## Progressive tests
"""this section enables to test various features of the code that were edited progressively, and to ensure that no conflits arise with previous work"""


# test 1 : parent class System
def test01() :
    syst = sample.core.System()
    syst.run()
    plt.plot(syst.times, syst.states)
    plt.title('Baseline simulation for System()')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()
    print('TEST 01 OK\n---------------------------------\n\n\n')
    return



# test 2 : input and run for BM
def test02() :
    syst = sample.core.BM(nbr=30, dt=0.01, n_step=1000)
    syst.run()
    syst.info()
    syst = sample.core.BM(nbr=30, dt=0.01, end_time=10)
    syst.info()
    syst.run()
    syst.info()
    st = syst.getState()
    plt.plot(np.arange(30), st)
    plt.show()
    print('TEST 02 OK\n---------------------------------\n\n\n')
    return


def test03() :
    syst = sample.core.BM(nbr=4, dt=0.01, end_time=10)
    syst.run()

    T = syst.getTimes()
    Ys = syst.getStates()
    Ys = np.transpose(np.array(Ys))
    Ys = sample.util.rescale(Ys)

    # three agents' time series
    sample.util.display(T, Ys[0], name='3 series', color='r')
    sample.util.display(T, Ys[1], name='3 series', color='g')
    sample.util.display(T, Ys[2], name='3 series', color='b', ylabel='rescaled weight')

    # all agents time serie
    sample.util.displayAll(T, Ys, name='all series', ylabel='rescaled weight')

    # all time series, cumulated
    sample.util.displayAll(T, sample.util.cumul(Ys), name='all series cumul', ylabel='rescaled weight (cumulative)')

    # all wealth hisotgram, cumulated
    N = np.shape(Ys)[0]
    sample.util.displayAll(np.arange(N), np.transpose(sample.util.cumul(Ys)), name='all histograms cumul', ylabel='rescaled weight (cumulative)')

    # rescaled cov matrix
    sample.util.displayMat(sample.util.rcov(Ys), name='cov matrix')

    # autocorrelation
    sample.util.display(T, sample.util.rautocorr(Ys[0]), name='autocorr', color='r')
    sample.util.display(T, sample.util.rautocorr(Ys[1]), name='autocorr', color='g')
    sample.util.display(T, sample.util.rautocorr(Ys[2]), name='autocorr', color='b', ylabel='autocorrelation')

    # autocovariance
    sample.util.display(T, sample.util.rautocov(Ys[0]), name='autocov', color='r')
    sample.util.display(T, sample.util.rautocov(Ys[1]), name='autocov', color='g')
    sample.util.display(T, sample.util.rautocov(Ys[2]), name='autocov', color='b', ylabel='autocovariance')

    # variogram
    sample.util.display(T, sample.util.rvariogram(Ys[0]), name='variogram', color='r')
    sample.util.display(T, sample.util.rvariogram(Ys[1]), name='variogram', color='g')
    sample.util.display(T, sample.util.rvariogram(Ys[2]), name='variogram', color='b', ylabel='variogram')

    # Y_2 index
    sample.util.display(T, sample.util.Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # histogram
    sample.util.displayHist(sample.util.Y2(Ys), name='Y2_hist', ylabel='$Y_2$ index')

    plt.show()
    print('TEST 03 OK\n---------------------------------\n\n\n')
    return







def test04() :
    # testing largeBM instanciation
    syst = sample.core.largeBM(nbr=30, dt=0.01, n_step=1000)
    syst.run()
    syst.info()
    syst = sample.core.largeBM(nbr=30, dt=0.01, end_time=10)
    syst.info()
    syst.run()
    syst.info()
    st = syst.getState()
    plt.plot(np.arange(30), st)
    plt.show()
    print('TEST 04 OK\n---------------------------------\n\n\n')
    return



def test05() :
    # testing largeBM
    syst = sample.core.largeBM(nbr=4, dt=0.01, end_time=20, noise_inpt=(1., 1.))
    syst.run()

    T = syst.getTimes()
    Ys = syst.getStates()
    Ys = np.transpose(np.array(Ys))

    # three agents' time series
    sample.util.display(T, Ys[0], name='3 series', color='r')
    sample.util.display(T, Ys[1], name='3 series', color='g')
    sample.util.display(T, Ys[2], name='3 series', color='b', ylabel='weight')

    # all agents time serie
    sample.util.displayAll(T, Ys, name='all series', ylabel='weight')

    # all time series, cumulated
    sample.util.displayAll(T, sample.util.cumul(Ys), name='all series cumul', ylabel='weight (cumulative)')

    # all wealth hisotgram, cumulated
    N = np.shape(Ys)[0]
    sample.util.displayAll(np.arange(N), np.transpose(sample.util.cumul(Ys)), name='all histograms cumul', ylabel='weight (cumulative)')

    # rescaled cov matrix
    sample.util.displayMat(sample.util.rcov(Ys), name='cov matrix')

    # autocorrelation
    sample.util.display(T, sample.util.rautocorr(Ys[0]), name='autocorr', color='r')
    sample.util.display(T, sample.util.rautocorr(Ys[1]), name='autocorr', color='g')
    sample.util.display(T, sample.util.rautocorr(Ys[2]), name='autocorr', color='b', ylabel='autocorrelation')

    # autocovariance
    sample.util.display(T, sample.util.rautocov(Ys[0]), name='autocov', color='r')
    sample.util.display(T, sample.util.rautocov(Ys[1]), name='autocov', color='g')
    sample.util.display(T, sample.util.rautocov(Ys[2]), name='autocov', color='b', ylabel='autocovariance')

    # variogram
    sample.util.display(T, sample.util.rvariogram(Ys[0]), name='variogram', color='r')
    sample.util.display(T, sample.util.rvariogram(Ys[1]), name='variogram', color='g')
    sample.util.display(T, sample.util.rvariogram(Ys[2]), name='variogram', color='b', ylabel='variogram')

    # Y_2 index
    sample.util.display(T, sample.util.Y2(Ys), name='Y2', ylabel='$Y_2$ index')

    # histogram
    sample.util.displayHist(sample.util.Y2(Ys), name='Y2_hist', ylabel='$Y_2$ index')

    plt.show()
    print('TEST 05 OK\n---------------------------------\n\n\n')
    return





# test NNNNN: ...
def test0NNNNN() :
    #...
    print('TEST 0NNNNN OK\n---------------------------------\n\n\n')
    return







## Dev



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
