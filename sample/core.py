"""core functions usefull for modelling the system or analysing results..."""


## Simulation

class System:
    """defines a dynamical system with a macrostate distribution function and a first-order linear master equation
    Inputs :
        int n_step : number of step that will be simulated
        float dt : time resolution
        float end_time : duration of the simulation (optionnal, has priority over the specified 'n_step')
        int dim : number of coordinates for a state
        str dyn : type of dynamic used for the master equation
        str noise : type of noise, '' if no noise"""

    def __init__(self, n_step=100, dt=0.1, end_time=-1., dim=5, dyn='ssr', noise='no noise', noise_inpt=(1.,10.)) :
        self.t = 0
        self.T = n_step
        self.dt = dt
        self.time = 0.
        if end_time > 0 :
            self.end_time = end_time
            self.T = int(end_time/dt)+1
        else :
            self.end_time = (self.T-1)*self.dt
        self.dim = dim
        self.state = np.zeros(self.dim, dtype='float')
        self.state[0] += 1.
        self.states = np.zeros((self.dim, self.T))
        self.states[:,0] = self.state # storage
        self.dyn = dyn
        self.noise = noise
        self.noise_inpt = noise_inpt
        self.J_0 = buildMatrix(dim=self.dim, dyn=self.dyn, param=1.)
        self.xi = np.zeros((self.dim, self.dim))
        self.xis = np.zeros((self.dim, self.dim, self.T)) # storage
        self.analysis = dict()
        self.running = False
        return

    def rebuildMatrix(self, p=1.) :
        """allows to input a parameter in the matrix"""
        self.J_0 = buildMatrix(dim=self.dim, dyn=self.dyn, param=p)
        return

    def info(self) :
        """prints info about the instance of class System"""
        if len(str(self.state)) > 36 :
            st = str(self.state)[:36]+'...'
        else :
            st = str(self.state)
        print("\n Object 'System' n°" + str(id(self)))
        print("-------------------------------------------------------------")
        print(' Simulation parameters\n     samples      : %d\n     length       : %d s\n     timestep     : %f s'%(self.T, int(self.end_time), self.dt))
        print(' System parameters\n     dynamic type : ' + self.dyn + '\n     noise type   : ' + self.noise + '\n     noise input  : ' + str(self.noise_inpt))
        print(' Current state\n     current step : %d\n     current time : %d s\n     state        : '%(self.t, int(self.time)) + st)
        print("-------------------------------------------------------------\n\n")
        return

    def reset(self) :
        """erases all storage variables as well as System.state"""
        self.t = 0
        self.time = 0.
        self.state = np.zeros(self.dim, dtype='float')
        self.state[0] += 1.
        self.states = np.zeros((self.dim, self.T))
        self.states[:,0] = self.state # storage
        self.xi = np.zeros((self.dim, self.dim))
        self.xis = np.zeros((self.dim, self.dim, self.T)) # storage
        self.analysis = dict()
        self.running = False
        return

    def doStep(self) :
        """forwards the state by dt
        before System.doStep() : System.t is the index of input variables (state(t), J(t)...)
        after System.doStep()  : System.t is the index of output variables (state(d+dt))
        (same for System.time)"""
        # setting the matrix
        if self.noise != 'no noise' :
            self.xi = genNoise(dim=self.dim, rule=self.noise, inpt=self.noise_inpt)
            self.xis[:,:,self.t] = self.xi # storage
            J = self.J_0 + self.xi
        else :
            J = self.J_0
        # master equation
        self.state += self.dt*np.dot(J, self.state) # driving equation : state(t+dt) = state(t) + dt*J(t)*state(t)
        # data
        self.states[:,self.t] = self.state # storage
        self.t += 1
        self.time += self.dt
        return

    def run(self) :
        """calculates all states from n°1 to n°T-1 (n°0 is set by default)
        calls a thread loop that shows progress if long simulation"""
        start = time.time()
        # parallel thread
        delay = 1.
        timer = RepeatTimer(delay, progress, (self,))
        timer.start()
        # main thread
        self.running = True
        #self.t = 1 # !! self.t représente le temps présent à l'entrée dans System.doStep() !!
        while self.t < self.T :
            self.doStep()
        self.running = False
        timer.cancel()
        delta = time.time() - start
        print('System.run() exit, runtime = ' + '{:.3f}'.format(delta) + ' s')
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
        plt.figure()
        plt.bar(Xs, self.state, width=wdth)
        plt.xlabel('state coordinates')
        plt.ylabel('density (not norm.)')
        # placing ticks on axes
        if self.dim <= 15 and not log :
            plt.xticks(Xs)
        #test plt.show()
        return

    def animateState(self, log=False) :
        """plots all the states stored in self.states in an animation
        the scale for y-axes is normalized with the mean density of state (not always 1 for non-stochastic matrices)"""
        # figure instanciation
        fig = plt.figure(figsize=(12,8))
        # fancying axes
        Xs = np.arange(self.dim)
        if self.dim <= 15 and not log :
            plt.xticks(Xs)
        plt.xlabel('agent')
        plt.ylabel('rescaled wealth')
        plt.ylim(0,self.dim)
        # setting an eventual log scale
        wdth = np.zeros(self.dim)+0.5
        if log :
            Xs += 1
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(10**-(self.dim/10),self.dim) # uncomment this for proper scale definition
            wdth = 0.05*Xs
        # plotting th first image
        mean = np.sum(self.state)/self.dim
        bars = plt.bar(Xs, self.state/mean, width=wdth)
        txt = plt.text(self.dim/2, self.dim/2, '', backgroundcolor=(1., 1., 1., 0.5))
        # animating function (syst is given by animation.FuncAnimation())
        def animate(t, syst) :
            t = t%syst.T
            state = syst.states[:,t]
            mean = np.sum(state)/self.dim
            for i in range(self.dim) :
                bars[i].set_height(state[i]/mean)
                sfx = ' out of ' + "{:.2f}".format(self.end_time) + ' s'
            tm = 't = ' + "{:.2f}".format(self.dt*t) + sfx
            txt.set_text(tm)
            return bars.patches + [txt]
        # display
        global ani # otherwise the 'ani' variable (existing locally only) is dumped after the 'return'
        ani = animation.FuncAnimation(fig, animate, frames=1000, fargs=(self,), interval=40, blit=True)
        plt.show()
        return

    def plotMatrix(self) :
        """plots the driving matrix stored in self.J_0"""
        # show matrix
        plt.figure()
        ax = plt.subplot()
        im = ax.matshow(self.J_0)
        # set colorbar (scale) next to matrix
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        # show
        #test plt.show()
        return





## Analysis

class BMtoolkit:
    """all tool used to analyse a Bouchau-Mézard model
    Class variables :
        ...
    Input :
        ...
    BMtoolkit.data codenames :
        'av_w' : time sequence of all averaged weights (average wealth in Bouchaud-Mézard),
        'n_av_w' = the averaged weights, normalized by the (observed) exponential growth at each time-step,
        'max_w'  = the maximum weight,
        'min_w'  = the minimum,
        'rsc_states' = syst.states rescaled at each timestep by the average weight over agents (must be a stationnary process)
        ''       = ..."""

    def __init__(self) :
        self.syst = System()
        self.data = dict()
        return

    def load(self, syst) :
        """loads a 'System' object as the input source"""
        self.syst = syst
        return

    def maxW(self) :
        """calculates the max weight for each time t and store the resulting time-sequence in BMtoolkit.data['maw_W']"""
        self.data['max_W'] = np.amax(self.syst.states, axis=0)
        return

    def minW(self) :
        """calculates the min weight for each time t and store the resulting time-sequence in BMtoolkit.data['min_W']"""
        self.data['min_W'] = np.amin(self.syst.states, axis=0)
        return

    def ratio(self) :
        """calculates the ratio max/min for each timestep, stores data in BMtoolkit.data['ratio']"""
        self.data['ratio'] = np.amax(self.syst.states, axis=0)/np.amin(self.syst.states, axis=0)
        return

    def avW(self) :
        """will average the *weights* along all states for each time t and store the resulting time-sequence in BMtoolkit.data['av_w']"""
        self.data['av_w'] = np.average(self.syst.states, axis=0)
        return

    def diff(self, key='') :
        """differentiate a time series over 1 timestep
        stores the result as self.data[key+'_diff']"""
        try:
            Ys = self.data[key]
        except:
            print("ERROR - BMtoolkit.diff() - BMtoolkit.data['av_w'] not assigned !")
            return
        L = np.shape(Ys)[0]
        self.data[key+'_diff'] = Ys[1:]-Ys[:L-1]
        return

    def normAvW(self) :
        """rescales the average-weight time sequence by deviding by its analytic expectancy exp((m+sim**2)t)
        stores the data in self.data['n_av_w']"""
        # getting input
        T = self.syst.T
        dt = self.syst.dt
        end_time = self.syst.end_time
        try:
            Ys = self.data['av_w']
        except:
            print("ERROR - BMtoolkit.normAvW() - BMtoolkit.data['av_w'] not assigned !")
            return
        # naively claculating growth rate
        delta_t = (T-1)*dt
        growth = self.data['av_w'][-1]/self.data['av_w'][0]
        rate = np.log(growth)/delta_t
        A0 = self.data['av_w'][0]
        # normalizing
        Ts = np.linspace(0., end_time, T)
        Es = A0*np.exp(rate*Ts)
        self.data['n_av_w'] = Ys/Es
        # WRONG LINE !!! self.data['n_Av_W'] = self.data['n_Av_W']/np.average(self.data['n_Av_W'])
        return

    def rescale(self) :
        """rescales each agent's weight time sequence by deviding by its the average over agents
        stores the data in self.data['rsc_states']

        TO CLEAN"""
        # getting input
        try:
            avs = self.data['av_w']
        except:
            print("ERROR - BMtoolkit.rescale() - BMtoolkit.data['av_w'] not assigned !")
            return
        avs = 1/avs # taking the inverse
        self.data['rsc_states'] = np.multiply(self.syst.states, avs)
        return

    def plotData(self, key='', log=False, ylabel='') :
        """plots a given time sequence stored in BMtoolkit.data"""
        # getting input
        end_time = self.syst.end_time
        try:
            Ys = self.data[key]
        except:
            print("ERROR - BMtoolkit.plotData() - BMtoolkit.data['" + key + "'] not assigned !")
            return
        L = np.shape(Ys)[0]
        dt = self.syst.dt
        # plotting
        fig = plt.figure()
        Ts = np.linspace(0., (L-1)*dt, L)
        if log :
            plt.yscale('log')
        plt.plot(Ts, Ys)
        plt.xlabel('time (s)')
        plt.ylabel(ylabel)
        return


    def plotHist(self, key='', ylabel='', log=False) :
        """plots the histogram of the specified time sequence"""
        # getting input
        #end_time = self.syst.end_time
        #L = np.shape(Ys)[0]
        #dt = self.syst.dt
        try:
            Ys = self.data[key]
        except:
            print("ERROR - BMtoolkit.plotHist() - BMtoolkit.data['" + key + "'] not assigned !")
            return
        # handling log-scale
        if log :
            m, M = np.amin(Ys), np.amax(Ys)
            em, eM = np.log10(m), np.log10(M)
            bins = np.logspace(em,eM, num=100)
            plt.xscale('log')
            plt.yscale('log')
        else :
            bins = 100
        # plotting
        hist = plt.hist(Ys, bins=bins)
        plt.ylabel(ylabel)
        return













class SSRtoolkit:
    def __init__(self, dim=5) :
        self.dim = dim
        return





