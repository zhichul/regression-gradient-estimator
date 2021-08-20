from noisy_gradient.utils import nd_riemann_sum
from mpl_toolkits import mplot3d
from tqdm import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.rint(0.1 * (x[0] ** 2 - x[1] ** 2))

def l(x, scale=1):
    dist = st.multivariate_normal(x, cov= scale * np.eye(2))
    return nd_riemann_sum(lambda X: f(X) * dist.pdf(X.transpose((1,2,0))), (-10, -10), (10, 10), dx=0.05)

def gradient_estimator(x, noise):
    _, M = noise.shape
    fs = f(x[:,None] + noise)
    A = np.zeros((3, M))
    A[0, :] = 1 # bias feature
    A[1:, :] = noise
    w = np.linalg.inv(A @ A.T) @ A @ fs
    return w[1:]

def main():
    lbound, rbound = -5, 5
    resolution = 20
    scale = 1
    nsamples = 100_000

    points = np.linspace(lbound, rbound, resolution)
    xs = np.meshgrid(points, points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot f(x)
    surf = ax.plot_wireframe(xs[0], xs[1], f(xs), label="f(x)", color="black")
    # surf._facecolors2d = surf._facecolor3d
    # surf._edgecolors2d = surf._edgecolor3d

    # plot E[f(x)]
    surf = ax.plot_surface(xs[0], xs[1], np.array([[l([axis[i,j] for axis in xs], scale=scale) for j in range(xs[0].shape[1])] for i in tqdm(range(xs[0].shape[0]))]), label="E[f(x)]", color="darkorange")
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # plot ∇E[f(x)]
    eps = 1e-4
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    g1 = np.array([[(l(np.array([axis[i,j] for axis in xs]) + eps * e1, scale=scale) - l(np.array([axis[i,j] for axis in xs])- eps * e1, scale=scale)) / (2*eps) for j in range(xs[0].shape[1])] for i in tqdm(range(xs[0].shape[0]))])
    g2 = np.array([[(l(np.array([axis[i,j] for axis in xs]) + eps * e2, scale=scale) - l(np.array([axis[i,j] for axis in xs])- eps * e2, scale=scale)) / (2*eps) for j in range(xs[0].shape[1])] for i in tqdm(range(xs[0].shape[0]))])
    surf = ax.quiver(xs[0], xs[1], xs[0] * 0 - 4, g1, g2, g1 * 0, label="∇E[f(x)]", color="darkblue", alpha=0.3)

    # plot gradient estimator
    noise = np.random.multivariate_normal([0, 0], cov= scale * np.eye(2), size=nsamples).transpose(1,0)
    g = np.array([[gradient_estimator(np.array([axis[i,j] for axis in xs]), noise=noise) for j in range(xs[0].shape[1])] for i in tqdm(range(xs[0].shape[0]))])
    g1, g2 = g[:,:,0], g[:,:,1]
    surf = ax.quiver(xs[0], xs[1], xs[0] * 0 - 4, g1, g2, g1 * 0, linestyle=':', label="regression estimator", color="darkred")

    ax.view_init(29, -146)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()