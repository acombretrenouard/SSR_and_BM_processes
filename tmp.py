import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # for plt.colorbar axes positionning
import matplotlib.animation as animation






def buildMatrix(dim=10, dyn='uni', param=1.) :
    """builds a stochastic matrix, corresponding to the specified dynamic (with kwarg. 'dyn')
        uni --> uniformly in [0, 1[
        mfd --> mean field (all coef. equal to param/n_states, "param" is equivalent to "J")
        bin --> binomial (1 sample with probability param)
        pow --> power-tail law (Pareto here)
        nrm --> normal law
        ssr --> ssr process (with state no <-> energy)
        BGe --> Boltzmann-Gibbs equilibrium"""
    matrix = np.zeros((dim,dim), dtype=float)
    for j in range(dim) :
        s = 0
        for i in range(dim) :
            rate = 0.
            if   dyn=='mfd' : rate = param/dim
            elif dyn=='uni' : rate = np.random.uniform()
            elif dyn=='bin' : rate = np.random.binomial(1,param)
            elif dyn=='pow' : rate = np.random.pareto(param)
            elif dyn=='nrm' : rate = np.abs(np.random.normal()) # ! the coefficients must be postive !
            elif dyn=='ssr' :
                if i<j : rate = 1/j
                elif (i==dim-1 and j==0) : rate=1
            elif dyn=='BGe' : print('ERROR buildMatrix : dyn BGe not done yet')
            else :
                print('DynamicalRule.initialize ERROR : unknown distribution, uniform used instead\n Reminder :\n    uni --> uniformly in [0, 1[\n    bin --> binomial (1 sample)\n    pow --> power-tail law (Pareto here)\n    nrm --> normal law')
                rate = np.random.uniform()
            matrix[i,j] = rate
            if i != j : s += rate
        matrix[j,j] = - s # s is the sum of all other transition rates : the matrix is thus stochastic
    return matrix









class System:
    """defines a dynamical system with a macrostate distribution function and a first-order linear master equation"""

    def __init__(self) :
        self.t = 0
        self.T = 100
        self.dim = 5
        self.state = np.zeros(self.dim, dtype='float')
        self.state[0] += 1.
        self.states = np.zeros((self.dim, self.T))
        self.states[:,0] = self.state # storage
        self.dyn = 'ssr'
        self.noisy = False
        self.J_0 = buildMatrix(dim=self.dim, dyn=self.dyn, param=1.)
        self.noise = lambda : np.zeros((self.dim, self.dim))
        self.xi = np.zeros((self.dim, self.dim))
        self.xis = np.zeros((self.dim, self.dim, self.T)) # storage
        self.analysis = dict()
        return

    def reset(self) :
        """erases all storage variables as well as System.state"""
        self.t = 0
        self.state = np.zeros(self.dim, dtype='float')
        self.state[0] += 1.
        self.states = np.zeros((self.dim, self.T))
        self.states[:,0] = self.state # storage
        self.xi = np.zeros((self.dim, self.dim))
        self.xis = np.zeros((self.dim, self.dim, self.T)) # storage
        self.analysis = dict()
        return

    def doStep(self) :
        if self.noisy :
            self.xi = self.noise()
            self.xis[:,:,self.t] = self.xi # storage
            J = self.J_0 + self.xi
        else :
            J = self.J_0
        self.state += np.dot(J, self.state) # here dt = 1.
        self.states[:,self.t] = self.state # storage
        self.t += 1
        return

    def run(self) :
        self.t = 1
        while self.t < self.T :
            self.doStep()
        return

    def plotState(self, log=False) :
        """plots the current state (from self.state)"""
        Xs = np.arange(self.dim)
        wdth = np.zeros(self.dim)+0.5
        # setting an eventual log scale
        if log :
            Xs += 1
            plt.xscale('log')
            plt.yscale('log')
            wdth = 0.05*Xs
        # plotting
        plt.bar(Xs, self.state, width=wdth)
        plt.xlabel('state coordinates')
        plt.ylabel('density (not norm.)')
        # placing ticks on axes
        if self.dim <= 15 and not log :
            plt.xticks(Xs)
        plt.show()
        return

    def animateState(self, log=False) :
        """plots all the states stored in self.states in an animation"""
        fig = plt.figure()
        Xs = np.arange(self.dim)
        wdth = np.zeros(self.dim)+0.5
        # setting an eventual log scale
        if log :
            Xs += 1
            plt.xscale('log')
            plt.yscale('log')
            wdth = 0.05*Xs
        # plotting
        mean = np.sum(self.state)/self.dim
        bars = plt.bar(Xs, self.state/mean, width=wdth)
        print(mean, self.dim)
        txt = plt.text(0.5*self.dim, 1, '')
        plt.xlabel('state coordinates')
        plt.ylabel('density (not norm.)')
        # placing ticks on axes
        if self.dim <= 15 and not log :
            plt.xticks(Xs)

        def animate(t, syst) :
            #if t >= self.T :
            #    print('WARNING - System.animateState() loc. fct. animate() : index t out of range (%d >= %d)'%(t,self.T))
            #    return
            t = t%syst.T
            print(t)
            state = syst.states[:,t]
            mean = np.sum(state)/self.dim
            for i in range(self.dim) :
                bars[i].set_height(state[i]/mean)
            txt.set_text(str(t))
            return bars.patches + [txt]

        # tests
        #print(type(txt), txt)
        #print(type(bars), bars)
        #print(type(bars.patches), bars.patches)
        ani = animation.FuncAnimation(fig, animate, fargs=(self,), interval=200, blit=True)

        plt.show()
        return

    def plotMatrix(self) :
        """plots the driving matrix stored in self.J_0"""
        # show matrix
        ax = plt.subplot()
        im = ax.matshow(self.J_0)
        # set colorbar (scale) next to matrix
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        # show
        plt.show()
        return



