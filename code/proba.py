import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt


def get_cum_proba(N, M):
    n = np.arange(N)

    proba = []

    for k in n:
        rv = hypergeom(N, M, k)
        proba.append(rv.pmf(0) * M / (N - k))

    proba = np.array(proba)
    cum_proba = np.array([proba[:i].sum() for i in n])

    return cum_proba


def plot_proba(N=1000):
    n = np.arange(N)
    p1 = get_cum_proba(N, N // 100)
    p5 = get_cum_proba(N, N // 20)
    p10 = get_cum_proba(N, N // 10)

    # plt.plot(n, proba)
    plt.plot(n, p1)
    plt.plot(n, p5)
    plt.plot(n, p10)
    plt.plot((0, N), (0.95, 0.95), 'r', linestyle='--')
    plt.scatter((np.where(p1 > 0.95)[0][0], np.where(p5 > 0.95)[0][0], np.where(p10 > 0.95)[0][0]), (0.95, 0.95, 0.95), color='r', zorder=2)
    plt.xlim([0, N])
    plt.show()


def plot_contour():
    nb_points = 5
    N = np.linspace(10, 10000, nb_points, dtype=np.int)
    M = np.linspace(1, 3000, nb_points, dtype=np.int)

    z = np.zeros((nb_points, nb_points))
    for i, n in enumerate(N):
        for j, m in enumerate(M):
            if m >= n:
                continue
            p = get_cum_proba(n, m)
            z[i, j] = np.where(p > 0.5)[0][0]

    Nm, Mm = np.meshgrid(N, M)
    z = z.T

    print(z.max())

    plt.contour(Nm, Mm, z)
    plt.show()

# plot_contour()
plot_proba(600)
