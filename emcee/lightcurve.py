from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.special import gammaln
from time import time as timer
import numpy as np
import emcee as mc

table = fits.open("kplr011904151-2010265121752_llc.fits")
tab = table[1]
data = Table(tab.data)

flux_orig = data['PDCSAP_FLUX'][:1325]
time_orig = data['TIME'][:1325]
ind_nan = [i for i in range(len(flux_orig)) if str(flux_orig[i]) == "nan"]
flux = [flux_orig[i] for i in range(len(flux_orig)) if i not in ind_nan]
time = [time_orig[i] for i in range(len(time_orig)) if i not in ind_nan]

# plot useful data
plt.figure(figsize=(20,5))
plt.plot(time, flux, "k:")
plt.xlim(539,567)
plt.ylim(541150,541750)
"""plt.show()"""

f_mu = 541500
f_sig = 100

df_mu = 2.77e-4
df_sig = 0.3e-4

p_mu = 0.84
p_sig = 0.05

tt_mu = 0.147
tt_sig = 0.05

tf_mu = 0.0882353
tf_sig = 0.005

off_mu = -0.05
off_sig = 0.005

priors = [(f_mu, f_sig), (df_mu, df_sig), (p_mu, p_sig), (tt_mu, tt_sig), (tf_mu, tf_sig), (off_mu, off_sig)]


def step(time, f, df, p, tt, tf, off=0):
    """
    Flux, from a uniform star source with single orbiting planet, as a function of time
    :param time: 1D array, input times
    :param f: unobstructed flux, max flux level
    :param df: ratio of obscured to unobscured flux
    :param p: period of planet's orbit
    :param tt: total time of transit
    :param tf: time during transit in which flux doesn't change
    :param off: time offset, a value of 0 means the transit begins immidiately
    :return: Flux from the star
    """
    y = []
    h = f*df*tt/(tt-tf)
    grad = 2*h/tt
    for i in time:
        j = (i + off) % p
        if j < tt:
            # transit
            if j/tt < 0.5:
                # first half of transit
                val = f - grad*j
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
            else:
                # last half of transit
                val = (grad*j) - 2*h + f
                if val < f*(1 - df):
                    y.append(f*(1 - df))
                else:
                    y.append(val)
        else:
            # no transit
            y.append(f)
    return y


def loglike(theta, obs, times):
    f_like, df_like, p_like, tt_like, tf_like, off_like = theta
    lmbda = np.array(step(times, f_like, df_like, p_like, tt_like, tf_like, off=off_like))
    n = len(obs)
    a = np.sum(gammaln(np.array(obs)+1))
    b = np.log(lmbda)*np.sum(obs)
    return -n*lmbda - a + b


def logprior(theta):
    lprior = 0
    for i in range(len(priors)):
        lprior -= 0.5*((theta[i] - priors[i][0]) / priors[i][1])**2
    return lprior


def logposterior(theta, obs, times):
    lprior = logprior(theta)
    if not np.isfinite(lprior):
        return -np.inf
    return lprior + loglike(theta, obs, times)


Nens = 100
inisamples = np.array([np.random.normal(priors[i][0], priors[i][1], Nens) for i in range(len(priors))]).T
ndims = inisamples.shape[1]

Nburn = 500
Nsamples = 500

loglike.ncalls = 0

sampler = mc.EnsembleSampler(Nens, ndims, logposterior, args=[flux, time])

t0 = timer()
sampler.run_mcmc(inisamples, Nsamples+Nburn)
t1 = timer()

samples_emcee = sampler.chain[:, Nburn:, :].reshape((-1, ndims))

