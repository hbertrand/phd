import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve


class StationaryKernel(object):
    def __init__(self):
        pass

    def compute(self, a):
        pass


class _StationaryDeltaKernel(StationaryKernel):
    def compute(self, a):
        return int(a == 0.)


class _StationarySquaredExponentialKernel(StationaryKernel):
    def __init__(self, l):
        super(StationaryKernel, self).__init__()

        self.l = l

    def compute(self, a):
        return np.exp(- (np.linalg.norm(a) ** 2) / (2 * self.l * self.l))


class _StationaryMatern52Kernel(StationaryKernel):
    def __init__(self, l):
        super(StationaryKernel, self).__init__()

        self.l = l

    def compute(self, a):
        r2 = (np.linalg.norm(a) ** 2) / (self.l * self.l)
        return (1 + np.sqrt(5 * r2) + (5 / 3) * r2) * np.exp(-np.sqrt(5 * r2))


class Kernel(object):
    def __init__(self):
        pass

    def compute(self, a, b):
        pass


class LinearKernel(Kernel):
    def compute(self, a, b):
        return np.dot(a.T, b)


class PolynomialKernel(Kernel):
    def __init__(self, p):
        super(Kernel, self).__init__()

        self.p = p

    def compute(self, a, b):
        return np.dot(a.T, b) ** self.p


class SquaredExponentialKernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l

    def compute(self, a, b):
        return np.exp(- (np.linalg.norm(a - b) ** 2) / (2 * self.l * self.l))


class Matern52Kernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l

    def compute(self, a, b):
        r2 = (np.linalg.norm(a - b) ** 2) / (self.l * self.l)
        return (1 + np.sqrt(5 * r2) + (5 / 3) * r2) * np.exp(-np.sqrt(5 * r2))


class OrnsteinUhlenbeckKernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l

    def compute(self, a, b):
        r = np.linalg.norm(a - b)
        return np.exp(-r/self.l)


class PeriodicKernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l

    def compute(self, a, b):
        return np.exp(- (2 * np.sin((a - b) / 2) ** 2) / (self.l * self.l))


class DeltaKernel(Kernel):
    def compute(self, a, b):
        return int(a == b)


class NeuralNetworkKernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l

    def compute(self, a, b):
        n = 1 + a.shape[0]
        sigma = np.identity(n)
        sigma[range(n), range(n)] = self.l

        an = np.append([1.], a, 0)
        bn = np.append([1.], b, 0)
        denominator = np.sqrt((1 + 2 * np.dot(an.T, np.dot(sigma, an))) * (1 + 2 * np.dot(bn.T, np.dot(sigma, bn))))
        res = (2. / np.pi) * np.arcsin((2 * np.dot(an.T, np.dot(sigma, bn))) / denominator)

        return res


class LocallyStationarySquaredExponentialKernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l
        self.k1 = _StationarySquaredExponentialKernel(l).compute
        self.k2 = _StationarySquaredExponentialKernel(l).compute
        # self.k2 = _StationaryDeltaKernel().compute

    def compute(self, a, b):
        return self.k1((a + b) / 2.) * self.k2(a - b)


class LocallyStationaryMatern52Kernel(Kernel):
    def __init__(self, l):
        super(Kernel, self).__init__()

        self.l = l
        self.k1 = _StationaryMatern52Kernel(l).compute
        self.k2 = _StationaryMatern52Kernel(l).compute
        # self.k2 = _StationaryDeltaKernel().compute

    def compute(self, a, b):
        return self.k1((a + b) / 2.) * self.k2(a - b)


class GaussianProcess(object):
    def __init__(self, kernel, sigma=1e-10, random_state=None):
        self.kernel = kernel.compute

        self.sigma = sigma

        if random_state is None:
            self.rs = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rs = np.random.RandomState(random_state)
        else:
            self.rs = random_state

        self.x = np.array([])
        self.y = np.array([])
        self.n = 0
        self.l = np.array([])
        self.alpha = np.array([])

    def fit(self, x, y):
        self.n = x.shape[0]
        self.x = x
        self.y = y

        k = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                k[i, j] = self.kernel(x[i], x[j])

        self.l = cholesky(k + self.sigma * np.identity(self.n), lower=True)
        # self.alpha = solve(self.l.T, solve(self.l, y))
        self.alpha = cho_solve((self.l, True), y)

    def predict(self, x, return_cov=False, return_std=False):
        n_samples_test = x.shape[0]
        k_star = np.zeros((self.n, n_samples_test))
        for i in range(self.n):
            for j in range(n_samples_test):
                k_star[i, j] = self.kernel(self.x[i], x[j])

        k_star_star = np.zeros((n_samples_test, n_samples_test))
        for i in range(n_samples_test):
            for j in range(n_samples_test):
                k_star_star[i, j] = self.kernel(x[i], x[j])

        f_star = np.dot(k_star.T, self.alpha)

        if return_cov or return_std:
            if len(self.l):
                v = solve(self.l, k_star)
            else:
                v = np.array([])

            v_star = k_star_star - np.dot(v.T, v)

            if return_std:
                v_star = np.sqrt(np.diag(v_star))

            return f_star.squeeze(), v_star
        else:
            return f_star.squeeze()

    def likelihood(self):
        ll = -0.5 * np.dot(self.y.T, self.alpha) - np.diag(self.l).sum() - self.n / 2. * np.log2(2 * np.pi)
        ll = ll.squeeze()

        return ll

    def sample_y(self, x, n_samples=1):
        y_mean, y_cov = self.predict(x, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = self.rs.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [self.rs.multivariate_normal(y_mean[:, i], y_cov, n_samples).T[:, np.newaxis]
                         for i in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples


def pred_simple(nb_points=100, training_set=0.1):
    def f(x):
        return np.sin(x)
        # return (x / 10.) ** 2

    stepsize = 10. / nb_points
    x = np.arange(-5, 5, stepsize).reshape(-1, 1)
    np.random.shuffle(x)
    cutoff = int(x.shape[0] * training_set)
    x_train, x_test = x[:cutoff], x
    x_train = np.sort(x_train, axis=0)[::-1]
    x_test = np.sort(x_test, axis=0)[::-1]
    y_train, y_test = f(x_train), f(x_test)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = pred_simple()

    # kernel = LinearKernel()
    # kernel = PolynomialKernel(3)
    kernel = SquaredExponentialKernel(1)
    # kernel = Matern52Kernel(1)
    # kernel = LocallyStationarySquaredExponentialKernel(1)
    # kernel = LocallyStationaryMatern52Kernel(1)
    # kernel = DeltaKernel()
    # kernel = PeriodicKernel(1)
    # kernel = NeuralNetworkKernel(1)
    gp = GaussianProcess(kernel, sigma=0.001)
    gp.fit(x_train, y_train)

    # f_star, v_star, ll = gp.predict(x_test)
    f_star, cov = gp.predict(x_test, return_cov=True)
    v_star = np.sqrt(np.diag(cov))

    y_samples = np.random.multivariate_normal(f_star, cov, 3).T

    plt.scatter(x_train[:, 0], y_train, color='r', zorder=5)
    plt.plot(x_test[:, 0], y_test[:, 0], 'r')
    plt.plot(x_test, f_star, 'b-', label=u'Prediction')
    plt.plot(x_test, y_samples, 'k-', label=u'Sample')
    plt.fill(np.concatenate([x_test, x_test[::-1]]),
             np.concatenate([f_star - 1.9600 * v_star, (f_star + 1.9600 * v_star)[::-1]]),
             alpha=.4, fc='b', ec='None', label="95% prediction interval")
    plt.show()
