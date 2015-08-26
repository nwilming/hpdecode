from numpy import *
# Requires scipy 0.14.0, consider anaconda to install it
from scipy.stats import multivariate_normal, poisson

'''
Example of bayesian reconstruction for place cell data
(with fake data of course)

'''


def place_field(pos, covariance, firing_rate=10,baseline=1):
    '''
    Creates a 2D Gaussian place field with center pos and
    covariance matrix. The max is scalled to firing_rate.
    Baseline gives the baseline firing rate.
    '''
    mv = multivariate_normal(pos, covariance)
    scale_constant = mv.pdf(pos)
    def pdf(arena):
        fr = firing_rate*mv.pdf(arena)/scale_constant + baseline
        try:
            fr[fr>firing_rate]=firing_rate
        except TypeError:
            if fr>firing_rate:
                fr = firing_rate
        return fr
    return pdf

def setup():
    '''
    Setup a standard arena (10x10) and standard place fields.
    '''
    pfield_centers = [(x,y) for x in arange(0,11) for y in arange(0,11)]
    pfields = [place_field(array(pos), [[.5,.0],[0,.50]]) for pos in pfield_centers]

    aX, aY = meshgrid(linspace(0,10,100), linspace(0,10,100))
    arena = empty(aX.shape+(2,))
    arena[:,:,0], arena[:,:,1] = aX,aY
    return pfields, arena, aX, aY, array(pfield_centers)

def simulate_spikes(pfields, rx, ry):
    '''
    Compute firing rate for each neuron given place field center
    and sample number of observed spikes in one time unit.
    '''
    rates = []
    obs_spikes = []
    for n, pfield in enumerate(pfields):
        rate = pfield((rx,ry))
        spikes = poisson.rvs(rate)
        rates.append(rate)
        obs_spikes.append(spikes)
    return rates, obs_spikes

def prior(arena):
    '''
    Compute log prior, either with two step position
    dependent, or just uniform.

    This would be the thing to change to include the two step prior.
    '''
    return  log(arena[:,:,0]/prod(arena.shape[:2]))


def likelihood(pfields, spikes, arena):
    '''
    Compute the log likelihood of observing number of spikes given
    in list spikes as a function of each place field.
    '''
    acc = 0*arena[:,:,0].flatten()
    for spikes, pfield in zip(spikes, pfields):
        rate = pfield(arena)
        p_n_x = poisson.pmf(spikes,rate.flatten())
        acc += log(p_n_x)
    return acc.reshape(arena.shape[:2])

def decode_bayes(pfields, spikes, arena, last_pos=None):
    '''
    Use bayesian approach for decoding.
    '''
    return exp(likelihood(pfields, spikes, arena) + prior(arena))

def decode_directbasis(pfields, spikes, arena):
    '''
    Use direct basis approach for decoding.
    '''
    estimate = 0*arena[:,:,0]
    for nspikes, p in zip(spikes,pfields):
        estimate += p(arena)*nspikes
    return estimate


def visualize():
    '''
    Just some ugly plotting to make an animation for the lecture.
    '''
    import pylab as pp
    pp.ion()
    def remove_spines(axes):
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.spines['bottom'].set_color('none')
        axes.spines['left'].set_color('none')
    pfields, arena, aX, aY, centers = setup()
    theta = linspace(0,2*pi,250)
    x = cos(theta)*5 + 5
    y = sin(theta)*5 + 5
    points, estimates_bayes, estimates_direct = [],[], []
    for rx, ry in zip(x,y):
        handles = []
        rates, obs_spikes = simulate_spikes(pfields, rx,ry)
        pp.subplot(1,2,1)
        a = decode_bayes(pfields, obs_spikes, arena)
        pp.contour(aX,aY,a)
        pp.plot(centers[:,0],centers[:,1],'k+', alpha=.5)
        pp.hot()
        ey,ex = unravel_index(argmax(a), a.shape)
        estimates_bayes.append((aX[0,ex],aY[ey,0]))
        points.append((rx,ry))
        pp.plot(array(points)[:,0], array(points)[:,1], 'b-', alpha=0.5)
        pp.plot(rx,ry,'bo')
        pp.plot(aX[0,ex],aY[ey,0], 'ko')
        pp.plot(array(estimates_bayes)[:,0], array(estimates_bayes)[:,1], 'k-', alpha=0.5)
        pp.ylim([-.5,10.5])
        pp.xlim([-.5,10.5])
        remove_spines(pp.gca())

        pp.subplot(1,2,2)
        a = decode_directbasis(pfields, obs_spikes, arena)
        pp.contour(aX,aY,a)
        pp.plot(centers[:,0],centers[:,1],'k+', alpha=.5)
        pp.hot()
        ey,ex = unravel_index(argmax(a), a.shape)
        estimates_direct.append((aX[0,ex],aY[ey,0]))
        points.append((rx,ry))
        pp.plot(array(points)[:,0], array(points)[:,1], 'b-',alpha=.5)
        pp.plot(rx,ry,'bo')
        pp.plot(aX[0,ex],aY[ey,0], 'ko')
        pp.plot(array(estimates_direct)[:,0], array(estimates_direct)[:,1], 'k-', alpha=.5)
        pp.ylim([-.5,10.5])
        pp.xlim([-.5,10.5])
        remove_spines(pp.gca())
        yield(handles)
