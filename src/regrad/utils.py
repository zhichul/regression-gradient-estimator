import numpy as np

def riemann_sum(f, a, b, dx=0.01):
    xs = np.arange(a, b, dx)
    return np.sum(f(xs) * dx)

def nd_riemann_sum(f, a, b, dx=0.01):
    assert len(a) == len(b)
    nd = len(a)
    xs = [np.arange(ai, bi, dx) for ai, bi in zip(a,b)]
    xs = np.array(np.meshgrid(*xs))
    res = np.sum(f(xs)* (dx ** nd))
    return res
