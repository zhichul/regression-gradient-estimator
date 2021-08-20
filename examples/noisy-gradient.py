import code

import numpy as np
import pylab as pl
from scipy.integrate import quadrature
import scipy.stats as st

norm = st.norm
laplace = st.laplace



norm_sampler = np.random.normal
laplace_sampler = np.random.laplace

choices  = [(norm, norm_sampler), (laplace, laplace_sampler)]
dist, sampler = choices[0]

def f(x):
    return np.rint(2.75 *np.sin(x)) #np.rint(np.abs(x))#np.sign(x)#

def mc(x):
    return np.mean(f(x+noise))

def integrate(f, a, b):
    dx = .01
    xs = np.arange(a, b, dx)
    return np.sum(f(xs) * dx)

def quad(x):
    # print(scale)
    # print(x)
    d = dist(x, scale)
    a = -10  #d.ppf(.1)
    b = 10 #d.ppf(.9)
    func = lambda X: f(X) * d.pdf(X)
    return integrate(func, a, b) #quadrature(func, a, b, maxiter=1000)[0]

def estimate(x):
    fs = f(x+noise)

    A = np.zeros((2, M))
    A[1,:] = 1                # bias feature
    A[0,:] = noise

    # Hand roll regularized least squares
    [slope, intercept] = w = np.linalg.inv(A @ A.T + 0) @ A @ fs # 0*1/scale*np.eye(2)

    return slope

M = 100_000
scale = 0.5

noise = sampler(0,scale,size=M)

xs = np.linspace(-5, 5, 100)

#pl.plot(xs, [mc(x) for x in xs])
pl.plot(xs, f(xs), label='f(x)', c='k')
pl.plot(xs, [quad(x) for x in xs], label='E[f(x)]', c='darkorange')

eps = 1e-4
# pl.plot(xs, [(mc(x+eps)-mc(x-eps))/(2*eps) for x in xs], label='montecarlo')
pl.plot(xs, [(quad(x+eps)-quad(x-eps))/(2*eps) for x in xs], label='∇E[f(x)]', c='b', alpha=0.5)

pl.plot(xs, [estimate(x) for x in xs], linestyle=':', c='r', label='regression estimator')

print("done")
pl.legend(loc='best')
pl.show()