"""utility functions : build, in/out, etc..."""

import threading
import numpy as np

## Utilities


def buildMatrix(dim=5, dyn='uni', param=1.) :
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
                print('DynamicalRule.initialize() ERROR : unknown distribution, uniform used instead\n Reminder :\n    uni --> uniformly in [0, 1[\n    bin --> binomial (1 sample)\n    pow --> power-tail law (Pareto here)\n    nrm --> normal law')
                rate = np.random.uniform()
            matrix[i,j] = rate
            if i != j : s += rate
        matrix[j,j] = - s # s is the sum of all other transition rates : the matrix is thus stochastic
    return matrix

def genNoise(dim=5, rule='BMs', inpt=(1.,10.)) :
    """generates a random matrix of shape (dim, dim) according to a certain rule ('rule')
    kwarg 'inpt' is a tuple that can contain some necessary parameters (ex. for a gaussian noise : mean and variance)
    uni --> uniformly in [0, 1[
    bin --> binomial (1 sample)
    pow --> power-tail law (Pareto here)
    nrm --> normal law. INPUT : (mean,scale)=inpt
    BMs --> normal law. INPUT : as 'nrm' but scale=np.sqrt(2)*inpt[1]"""
    if rule=='nrm' :
        mean, sigm = inpt
        matrix = np.diag(np.random.normal(loc=mean, scale=sigm, size=dim))
    elif rule=='BMs' :
        mean, sigm = inpt
        matrix = np.diag(np.random.normal(loc=mean, scale=np.sqrt(2)*sigm, size=dim))
    elif rule=='uni' :
        matrix = np.diag(np.random.uniform(size=dim))
    elif rule=='bin' :
        matrix = np.diag(np.random.binomial(n=1, size=dim))
    elif rule=='pow' :
        matrix = np.diag(np.random.pareto(size=dim))
    elif False :
        print('Impossible !')
    else :
        print('ERROR - genNoise() - unknown noise rule, normal law used instead (mean=1., scale=10.)\nReminder :\n    uni --> uniformly in [0, 1[\n    bin --> binomial (1 sample)\n    pow --> power-tail law (Pareto here)\n    nrm --> normal law (scale=inpt[1]\n    BMs --> normal law where scale=np.sqrt(2)*inpt[1]')
        mean, sigm = 1.,10.
        matrix = np.diag(np.random.normal(loc=mean, scale=sigm, size=dim))
    return matrix

def progress(syst) :
    """fancy printing of progress in a run of a simulation"""
    if syst.running :
        pc = 100.*float(syst.t)/float(syst.T)
        print('Run in progess : ' + "{:.1f}".format(pc) + ' %')
    else :
        print('Not running')
    return

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
        return





