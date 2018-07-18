import numpy as np
import scipy.stats as ss

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from gaussian_process import GaussianProcess
from gaussian_process import SquaredExponentialKernel, NeuralNetworkKernel, DeltaKernel, PeriodicKernel, \
    Matern52Kernel, OrnsteinUhlenbeckKernel


def random_search(points='no', show=False):
    start, end = -5., 5.
    x = np.linspace(start, end, 1000)
    xx, yy = np.meshgrid(x, x)

    # z = xx ** 2 + 5 * xx + 2 * yy ** 2 + np.cos(xx)
    # zx = 20 * (x ** 2) + 250 / 3. + 2 * np.sin(5)
    # zy = 10 * (x ** 2 + 5 * x + np.cos(x)) + 250 / 3.

    def fx(a):
        return np.sin(a) + a / 5 + 5

    def fy(a):
        return (a ** 2) / 2 + np.cos(a) * a * 3

    zx = fx(x)
    zy = fy(x)
    z = fx(yy) + fy(xx)

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    fig = plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    nullfmt = NullFormatter()
    ax_scatter.xaxis.set_major_formatter(nullfmt)
    ax_scatter.yaxis.set_major_formatter(nullfmt)
    ax_histx.xaxis.set_major_formatter(nullfmt)
    ax_histx.yaxis.set_major_formatter(nullfmt)
    ax_histy.xaxis.set_major_formatter(nullfmt)
    ax_histy.yaxis.set_major_formatter(nullfmt)

    ax_scatter.xaxis.set_visible(False)
    ax_scatter.yaxis.set_visible(False)
    ax_histx.xaxis.set_visible(False)
    ax_histx.yaxis.set_visible(False)
    ax_histy.xaxis.set_visible(False)
    ax_histy.yaxis.set_visible(False)

    ax_scatter.contourf(xx, yy, z)
    ax_histx.plot(x, zy)
    ax_histx.fill_between(x, zy, -10000 + np.abs(zy) * 0., alpha=0.3)
    ax_histy.plot(zx, x)
    ax_histy.fill_betweenx(x, zx, -10000 + np.abs(zx) * 0., alpha=0.3)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    ax_histx.set_ylim((min(zx.min(), zy.min()), max(zx.max(), zy.max())))
    ax_histy.set_xlim((min(zx.min(), zy.min()), max(zx.max(), zy.max())))

    # Points
    if points == 'grid':
        x = np.linspace(-4, 4., 3)
        xp, yp = np.meshgrid(x, x)

        ax_scatter.scatter(xp, yp, s=100, c='k', zorder=5)
        ax_histx.scatter(x, fy(x), s=100, c='k', zorder=5)
        ax_histy.scatter(fx(x), x, s=100, c='k', zorder=5)
    elif points == 'random':
        # xp = np.random.uniform(start, end, 9)
        # yp = np.random.uniform(start, end, 9)
        xp = np.array([-4.3, -2.7, -1.6, -0.9, 0.2, 2.1, 2.5, 3.4, 4.6])
        yp = np.copy(xp)
        np.random.RandomState(5).shuffle(yp)

        ax_scatter.scatter(xp, yp, s=100, c='k', zorder=5)
        ax_histx.scatter(xp, fy(xp), s=100, c='k', zorder=5)
        ax_histy.scatter(fx(yp), yp, s=100, c='k', zorder=5)

    ax_scatter.set_xlim(ax_histx.get_xlim())
    ax_scatter.set_ylim(ax_histy.get_ylim())

    if show:
        plt.show()
    else:
        fig.patch.set_alpha(0.)
        fig.savefig('/mnt/disk2/bertrand/datasets/to_send/pp/rs_{0}.png'.format(points),
                    bbox_inches='tight', pad_inches=0)


def gaussian_process(kernel, nb_training_points=6, nb_samples=10, plot_dist=False, plot_gt=True, save=None):
    # Prepare training and test set
    def f(x):
        return 2 * np.sin(2*x) / x

    rs = np.random.RandomState(5)
    x = np.linspace(0.1, 10., 50)
    rs.shuffle(x)

    x_train = x[:nb_training_points]
    y_train = f(x_train)

    x_pred = np.linspace(-1, 12., 100)
    y = f(x_pred)

    # Fit and predict
    gp = GaussianProcess(kernel, random_state=rs)
    if nb_training_points > 0:
        gp.fit(x_train.reshape(-1, 1), y_train)

    y_pred, std_pred = gp.predict(x_pred.reshape(-1, 1), return_std=True)
    y_pred = y_pred.squeeze()

    std_pred += 1e-15  # Nobody likes 0

    # Configure plot settings
    color = sns.diverging_palette(15, 255, n=9, s=90, l=40)
    fig = plt.figure(figsize=(12, 4))
    sns.set_style("dark")
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    # Plot ground truth if required
    if plot_gt:
        plt.plot(x_pred, y, c=color[1], lw=3, label=u'Truth')

    # Plot mean and 95% prediction interval if required
    if plot_dist:
        plt.plot(x_pred, y_pred, c=color[8], lw=3, label=u'Prediction', zorder=4)
        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_pred - 1.9600 * std_pred, (y_pred + 1.9600 * std_pred)[::-1]]),
                 alpha=.4, fc=color[7], ec='None', label="95% prediction interval")

    # Plot required number of samples
    if nb_samples > 0:
        samples = gp.sample_y(x_pred.reshape(-1, 1), nb_samples)
        plt.plot(x_pred, samples)

    # Plot training set
    plt.scatter(x_train, y_train, facecolors=color[0], s=80, zorder=5)

    # More plot settings
    plt.xlim([-1, 12.])
    plt.ylim([-5, 5.])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.close()


def gaussian_process_ei():
    import seaborn as sns
    from sklearn.gaussian_process import GaussianProcessRegressor
    from matplotlib.animation import FuncAnimation

    def expected_improvement(target, y_pred, std_pred):
        v = target - y_pred
        u = v / std_pred
        return v * ss.norm.cdf(u) + std_pred * ss.norm.pdf(u)

    def f(x):
        return 2 * np.sin(2*x) / x
        # return np.sin(x*3) * x

    x = np.linspace(0.1, 10., 1000)
    y = f(x)

    x_train = np.array([3., 8.]).reshape(-1, 1)
    x_pred = x

    color = sns.diverging_palette(15, 255, n=9, s=90, l=40)
    fig = sns.plt.figure(figsize=(12, 4))
    sns.set_style("dark")
    ax = sns.plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    class UpdateAnim(object):
        def __init__(self, ax, x_train, x_pred):
            self.x_train_ori = x_train
            self.x_pred_ori = x_pred
            self.x_train = []
            self.x_pred = []

            self.ax = ax

            sns.plt.xlim([0.1, 10.])
            sns.plt.ylim([-5.2, 4.])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            y_tmp = np.zeros(x_pred.shape)
            self.std, = sns.plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                                     np.concatenate([y_tmp - 1.96, (y_tmp + 1.96)[::-1]]),
                                     alpha=.4, fc=color[7], ec='None', label="95% confidence interval",
                                     animated=True)
            self.pred_plot, = sns.plt.plot([], [], c=color[8], lw=3, label=u'Prediction', animated=True)
            self.ei_plot, = sns.plt.plot([], [], c=sns.xkcd_rgb["deep green"], lw=3, animated=True)
            self.points = sns.plt.scatter([], [], facecolors=color[0], s=80, animated=True, zorder=5)
            self.last_point = sns.plt.scatter([], [], facecolors=sns.xkcd_rgb["dark plum"], s=100, animated=True,
                                              zorder=5)
            self.marker = sns.plt.scatter([], [], marker='p', facecolors=sns.xkcd_rgb["very dark green"],
                                          s=120, animated=True, zorder=5)

        def init(self):
            sns.plt.plot(x, y, c=color[1], lw=3, label=u'Truth')

            self.x_train = self.x_train_ori
            self.x_pred = self.x_pred_ori

            return self.std, self.pred_plot, self.ei_plot, self.points, self.last_point, self.marker,

        def __call__(self, i):
            if i == 0:
                return self.init()

            if i % 10 == 0:
                y_train = f(self.x_train)

                gp = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=True)
                gp.fit(self.x_train, y_train)

                y_pred, std_pred = gp.predict(self.x_pred.reshape(-1, 1), return_std=True)
                y_pred = y_pred.squeeze()

                target = y_train.min()

                std_pred += 1e-15

                func_res = expected_improvement(target, y_pred, std_pred)

                std2, = plt.fill(np.concatenate([self.x_pred, self.x_pred[::-1]]),
                                 np.concatenate([y_pred - 1.9600 * std_pred, (y_pred + 1.9600 * std_pred)[::-1]]),
                                 alpha=.0, fc='r', ec='None')  # FIXME ugly dirty hack
                self.std.set_xy(std2.get_xy())

                self.pred_plot.set_data(self.x_pred, y_pred)
                self.ei_plot.set_data(self.x_pred, func_res * 5. - 5)
                self.points.set_offsets(np.concatenate([self.x_train[:-1], y_train[:-1]], axis=1))
                self.last_point.set_offsets([self.x_train[-1], y_train[-1]])
                self.marker.set_offsets([self.x_pred[func_res.argmax()], func_res.max() * 5. - 5])

                print(i, self.x_pred[func_res.argmax()], func_res.max(), target)
                self.x_train = np.append(self.x_train, [self.x_pred[func_res.argmax()]]).reshape(-1, 1)
                self.x_pred = np.delete(self.x_pred, func_res.argmax())

            return self.std, self.pred_plot, self.ei_plot, self.points, self.last_point, self.marker,

    ud = UpdateAnim(ax, x_train, x_pred)
    ani = FuncAnimation(fig, ud, frames=np.arange(200), init_func=ud.init, interval=100, blit=True, repeat=False)
    ani.save('/mnt/disk2/bertrand/datasets/to_send/pp/gp_ei.mp4', extra_args=['-vcodec', 'libx264'])

    sns.plt.show()
    sns.plt.close()


def gaussian_processes_marginal(show=False):
    import seaborn as sns
    from matplotlib.animation import FuncAnimation
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from scipy.stats import norm

    def f(x):
        return 2 * np.sin(2 * x) / x
        # return np.sin(x*3) * x

    rs = np.random.RandomState(5)
    x = np.linspace(0.1, 10., 50)
    rs.shuffle(x)

    sep = 6

    x_train = x[:sep]
    y_train = f(x_train)

    x_pred = np.linspace(-2, 12., 100)

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1., length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    if sep > 0:
        gp.fit(x_train.reshape(-1, 1), y_train)

    y_pred, std_pred = gp.predict(x_pred.reshape(-1, 1), return_std=True)
    y_pred = y_pred.squeeze()

    std_pred += 1e-15

    color = sns.diverging_palette(15, 255, n=9, s=90, l=40)
    fig = sns.plt.figure(figsize=(12, 4))
    sns.set_style("dark")
    ax = sns.plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    # sns.plt.plot(x_pred, y, c=color[1], lw=3, label=u'Truth')
    moving_palette = sns.color_palette("BuGn_r")
    current_std, = sns.plt.plot([], [], color=moving_palette[2], linewidth=3, animated=True)
    current_point = sns.plt.scatter([], [], facecolors=moving_palette[0], s=100, animated=True)

    current_normal, = sns.plt.plot([], [], color='k', linewidth=2, animated=True)

    def init():
        sns.plt.plot(x_pred, y_pred, c=color[8], lw=3, label=u'Prediction', zorder=4)
        sns.plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([y_pred - 1.9600 * std_pred, (y_pred + 1.9600 * std_pred)[::-1]]),
                     alpha=.4, fc=color[7], ec='None', label="95% confidence interval")

        sns.plt.scatter(x_train, y_train, facecolors=color[0], s=80, zorder=5)

        sns.plt.xlim([-2, 12.])
        sns.plt.ylim([-5, 5.])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        return current_std, current_point, current_normal,

    def update(frame):
        current_std.set_data([x_pred[frame], x_pred[frame]],
                             [y_pred[frame] - 1.9600 * std_pred[frame], y_pred[frame] + 1.9600 * std_pred[frame]])
        current_point.set_offsets([x_pred[frame], y_pred[frame]])

        x_pdf = np.linspace(-5, 5, 10000)
        p = norm.pdf(x_pdf, y_pred[frame], std_pred[frame])

        current_normal.set_data(p + x_pred[frame], x_pdf)

        return current_std, current_point, current_normal,

    ani = FuncAnimation(fig, update, frames=np.arange(len(x_pred)), init_func=init, interval=100, blit=True)
    ani.save('/mnt/disk2/bertrand/datasets/to_send/pp/gp.mp4', extra_args=['-vcodec', 'libx264'])

    if show:
        sns.plt.show()
    sns.plt.close()


def paper_vol_slice():
    from medisys_dl.dataset import hdf5_dict
    from keras.models import model_from_json

    # Load data
    dataset = hdf5_dict("/mnt/disk2/bertrand/datasets/hdf5/id_1411.hdf5")
    x = np.array(dataset['x'], dtype=np.uint16)
    n_s = x.shape[0]
    x = x.reshape((n_s, 1, 128, 128))
    y = np.array(dataset['y'], dtype=np.uint32)

    # Load best model
    dir_clf = "/mnt/disk2/bertrand/results/MRSearch/hyperopt/HCAPLS_512_4_V2_hyperband_gp_ei/model_292/"
    with open("{0}model.json".format(dir_clf), 'r') as f:
        classifier = model_from_json(f.read())
    classifier.load_weights("{0}weights.27.h5".format(dir_clf))
    classifier.compile(loss="categorical_crossentropy", optimizer='sgd')

    pred = classifier.predict(x)
    print(pred.shape)

    # entropy = np.sum(pred * np.log(pred + 1e-15), 1)
    # entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
    # y_pred = np.argmax(pred > 0.7, 1)
    # y_pred = np.array([y_pred[i] if entropy[i] > 0.5 else 4 for i in range(n_s)])

    dim_0, dim_1 = np.where(pred > 0.7)
    y_pred = np.ones(len(pred)) * 7
    for i in range(len(pred)):
        if i in dim_0:
            y_pred[i] = dim_1[np.where(dim_0 == i)[0][0]]
    y_pred = y_pred.astype(int)

    class_change = [np.where(y == 0)[0],
                    np.where(y == 2)[0],
                    np.where(y == 3)[0],
                    np.where(y == 4)[0],
                    np.where(y == 5)[0]]

    import seaborn as sns

    f, ax = sns.plt.subplots(2, 1)

    img = np.repeat(x[:, 0, :, 64].T, 5, 0)
    ax[1].matshow(img, zorder=1, cmap=plt.get_cmap("Greys_r"))
    ax[1].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
                      labelleft='off', labeltop='off')

    grad = sns.color_palette('bright')

    s = [np.ones(class_change[i].shape[0]) for i in range(len(class_change))]
    for i in range(len(class_change)):
        ax[1].fill_between(class_change[i], s[i] * -1., s[i] * 128. * 5., color=grad[i], alpha=0.3, zorder=2)

    for i in range(y_pred.shape[0]):
        if y_pred[i] < 6:
            ax[0].fill_between((i, i), (-1., -1.), (2., 2.), color=grad[y_pred[i]], alpha=0.5)

    for i in range(pred.shape[1]):
        ax[0].plot(np.arange(n_s), pred[:, i], label=['Head', 'Chest', 'Abdomen', 'Pelvis', 'Legs', 'Spine'][i])

    ax[0].set_xlim([0., n_s])
    ax[0].set_ylim([-0.05, 1.05])
    ax[0].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off')

    legend = ax[0].legend(loc='center left', frameon=True, shadow=True)
    legend.get_frame().set_facecolor("#DDDDDD")
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(10)
    sns.plt.tight_layout()
    f.subplots_adjust(hspace=0., wspace=0.)

    sns.plt.show()


def make_figures():
    # kernel = SquaredExponentialKernel(1.)
    # kernel = NeuralNetworkKernel([100., 100.])
    # kernel = DeltaKernel()
    # kernel = PeriodicKernel(.5)
    # kernel = Matern52Kernel(1.)
    # kernel = OrnsteinUhlenbeckKernel(1.)

    # GP prior
    gaussian_process(SquaredExponentialKernel(1.), 0, 5, True, False, save='figs/gp_prior.png')

    # GP posterior 1 point
    gaussian_process(SquaredExponentialKernel(1.), 1, 5, True, False, save='figs/gp_posterior_1_point.png')

    # GP posterior 6 point
    gaussian_process(SquaredExponentialKernel(1.), 6, 5, True, False, save='figs/gp_posterior_6_point.png')


if __name__ == '__main__':
    # random_search()
    # random_search('grid')
    # random_search('random')
    # gaussian_process_ei()

    # gaussian_processes_marginal(True)

    # paper_vol_slice()

    make_figures()
