"""test the functions progressively
!! needs to run 'sample/util.py' and 'sample/core.py' to run !!"""

def main() :
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
    return


## Tests

def test00() :
    print("test 00 : building a 'System' object")
    syst = System()
    print(syst.dim)
    print(syst.state)
    print(syst.J_0)
    print(syst.noise)
    print(syst.run)
    syst.plotMatrix()
    return


def test01() :
    print('test 1 : instanciation of class System')
    syst = System()
    a = syst.noise_inpt
    print(syst.noise_inpt)
    mat = genNoise(dim=4, rule='nrm')
    print(mat)
    return




def test02() :
    print('test 2 : System.doStep(), System.reset(), System.run(), syst.plotState()')
    plt.close('all')
    syst = System()
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
    syst = System()
    syst.run()
    syst.animateState()
    plt.show()
    return




def test04() :
    print('test 4 : inputs')
    plt.close('all')
    syst = System(dim=30, dt=0.01, n_step=1000)
    syst.info()

    syst = System(dim=30, dt=0.25, end_time=100)
    syst.info()
    syst.plotMatrix()
    syst.run()
    syst.animateState(log=False)
    plt.show()
    return




def test05() :
    print("test 5 : printing runtime (in System.run(), set 'delay' to 0.01)")
    syst = System(n_step=10000)
    syst.run()
    return





def test06() :
    print('test 6 : noise')
    plt.close('all')
    syst = System(dim=30, dt=0.01, end_time=100, dyn='mfd', noise='nrm', noise_inpt=(1.,10.))
    syst.run()
    syst.animateState()
    plt.show()
    return




def test07() :
    print('test 7 : BMtoolkit class, summing weights and plotting')
    plt.close('all')
    syst = System(dim=30, dt=0.025, end_time=99.976, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
    syst.info()
    syst.run()

    tlk = BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.plotData(log=True, key='av_w')
    plt.show()
    return




def test08() :
    print('test 8 : normalized average weight')
    plt.close('all')
    syst = System(dim=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(1., 10.))
    syst.run()
    syst.info()

    tlk = BMtoolkit()
    tlk.load(syst)
    tlk.avW()
    tlk.plotData(log=True, key='av_w', ylabel='average wealth')
    tlk.normAvW()
    tlk.plotData(log=False, key='n_av_w')
    plt.show()
    return





def test09() :
    print('test 9 : rescale function')
    syst = System(dim=5, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
    syst.run()

    tlk = BMtoolkit()
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
    syst = System(dim=5, dt=0.025, end_time=100, dyn='mfd', noise='abc', noise_inpt = (1,2,3))
    for i in range(3) :
        mat = genNoise(dim=syst.dim, rule=syst.noise, inpt=syst.noise_inpt)
        print(mat)
    return






def test11() :
    print('test 11 : time indexation')
    syst = System(n_step=6, dt=0.3)
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









## Dev





"""
# draft for two deltaHist() and plotDeltaHist() functions

syst = System(dim=30, dt=0.001, end_time=100, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = BMtoolkit()
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

syst = System(dim=30, dt=0.001, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = BMtoolkit()
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



syst = System(dim=30, dt=0.001, n_step=100000, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))

for i in range(3) :
    syst.reset()
    syst.run()
    tlk = BMtoolkit()
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

syst = System(dim=30, dt=0.001, n_step=100000, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))

for i in range(3) :
    syst.reset()
    syst.run()
    tlk = BMtoolkit()
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

syst = System(dim=300, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()

tlk = BMtoolkit()
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
syst = System(dim=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()


#
%matplotlib notebook

tlk = BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')


#
syst = System(dim=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
syst.run()

tlk = BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()
tlk.data['ag_0'] = tlk.data['rsc_states'][0,:]
tlk.plotData(key='ag_0', ylabel='rescaled weight of agent n°0')
tlk.data['ag_6'] = tlk.data['rsc_states'][6,:]
tlk.plotData(key='ag_6', ylabel='rescaled weight of agent n°6')






#
syst = System(dim=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.run()
# plotting histogram for ||J_0|| ~ 1 Hz

%matplotlib inline

tlk = BMtoolkit()
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

tlk = BMtoolkit()
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

syst = System(dim=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
syst.run()




#... continuing over last cell

%matplotlib inline

tlk = BMtoolkit()
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

syst = System(dim=30, dt=0.0001, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(0.3, 0.04))
syst.rebuildMatrix(p=10)
i1=0
i2=3
syst.J_0[i1,i2] += 100
syst.J_0[i2,i2] -= 100
syst.J_0[i2,i1] += 100
syst.J_0[i1,i1] -= 100
syst.run()







#...and we see the correlation

tlk = BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.rescale()

plotCov(tlk, inf=100000)









# experiment 3 : noise must surely lower correlations TO CONTINUE


syst = System(dim=30, dt=0.01, end_time=20, dyn='mfd', noise='BMs', noise_inpt=(1.,10.))
syst.rebuildMatrix(p=0.1)
syst.run()
syst.info()

tlk = BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.plotData(log=True, key='av_w')







syst = System(dim=30, dt=0.01, end_time=10, dyn='mfd', noise='BMs', noise_inpt=(0.1, 10.))
syst.run()
syst.info()

tlk = BMtoolkit()
tlk.load(syst)
tlk.avW()
tlk.plotData(log=True, key='av_w')

"""
