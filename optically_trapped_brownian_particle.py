import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner
plt.rcParams.update({'font.size': 13})

k_B = 1.38 * 1e-23
R = 1e-6
eta = 0.001
gamma = 6 * np.pi * eta * R
T = 300
k_x = 1e-12 / 1e-6
k_y = 2.5 * 1e-12 / 1e-6
n_timesteps = int(2e6)
n_timesteps_var = int(1e4)
delta_t = 0.01 * gamma / k_x
t_vector = np.linspace(0, int(n_timesteps * delta_t), n_timesteps)


def simulate_langevin(_position, _k, _gamma, _delta_t, _k_B, _T, _n_timesteps):
    for i in range(_n_timesteps - 1):
        _w = np.random.normal(loc=0, scale=1)
        _position[i + 1] = _position[i] - _k / _gamma * _position[i] * _delta_t + np.sqrt(2 * _k_B * _T * _delta_t /
                                                                                          _gamma) * _w
    return _position


def boltzmann(_x, _k, _k_B, _T):
    _U = 0.5 * _k * (_x * 1e-9) ** 2
    _p_boltzmann = np.exp(- _U / (_k_B * _T))
    return _p_boltzmann


def boltzmann_2D(_x, _y, _k_x, _k_y, _k_B, _T):
    _p_boltzmann_2d = np.zeros([len(_x), len(_y)])
    _const = - 1 / (2 * _k_B * _T)
    for i in range(len(_x)):
        for j in range(len(_y)):
            _p_boltzmann_2d[i, j] = np.exp(_const * (_k_x * _x[i] ** 2 + _k_y * _y[j] ** 2))
    return _p_boltzmann_2d


def boltzmann_2D2(_x, _y, _k_x, _k_y, _k_B, _T,):
    _const = - 1 / (2 * _k_B * _T)
    _p_boltzmann_2d2 = np.exp(_const * (_k_x * _x ** 2 + _k_y * _y ** 2))
    return _p_boltzmann_2d2


x_vector = np.zeros(n_timesteps)
x_vector = simulate_langevin(x_vector, k_x, gamma, delta_t, k_B, T, n_timesteps) * 1e9
y_vector = np.zeros(n_timesteps)
y_vector = simulate_langevin(y_vector, k_y, gamma, delta_t, k_B, T, n_timesteps) * 1e9



# p_boltzmann_2d = boltzmann_2D(x_vector * 1e-9, y_vector * 1e-9, k_x, k_y, k_B, T, n_timesteps_var)

min_x = np.min(x_vector)
max_x = np.max(x_vector)

n_x, bin_x = np.histogram(x_vector, bins=100, density=True)
x_bins = bin_x[:-1] + (bin_x[1] - bin_x[0]) / 2
n_x /= np.sum(n_x)

n_y, bin_y = np.histogram(y_vector, bins=100, density=True)
y_bins = bin_y[:-1] + (bin_y[1] - bin_y[0]) / 2
n_y /= np.sum(n_y)
p_boltzmann_x = boltzmann(x_bins, k_x, k_B, T)
p_boltzmann_x /= np.sum(p_boltzmann_x)

p_boltzmann_y = boltzmann(y_bins, k_y, k_B, T)
p_boltzmann_y /= np.sum(p_boltzmann_y)


def autocorrelation(_data, _n_steps):
    _autocorrelation = np.zeros(_n_steps)
    _mean_data = np.mean(_data)
    for k in range(_n_steps):
        _autocorrelation[k] = np.sum((_data[0] - _mean_data) * (_data[k] - _mean_data)) \
                              / np.sum((_data[0] - _mean_data) ** 2)
    return _autocorrelation


def autocorrelation_th(_t_vector_auto, _k_B, _T, _k, _gamma):
    _autocorrelation_th = k_B * _T / _k * np.exp(- _k * _t_vector_auto / _gamma)
    return _autocorrelation_th


x_auto = sm.tsa.acf(x_vector, nlags=n_timesteps)
y_auto = sm.tsa.acf(y_vector, nlags=n_timesteps)
t_vector_auto = np.linspace(0, 0.1, int(len(x_auto)))
x_auto_th = autocorrelation_th(t_vector, k_B, T, k_x, gamma)
x_auto_th /= x_auto_th[0]
y_auto_th = autocorrelation_th(t_vector, k_B, T, k_y, gamma)
y_auto_th /= y_auto_th[0]


def variance_th(_k_B, _T, _k):
    _sigma2 = _k_B * _T / _k
    return _sigma2


k_linspace = np.linspace(0.1, 10, 100) * 1e-12/1e-6
k_values = np.array([0.1, 0.2, 0.5, 0.8, 1, 2, 3, 4, 5, 7, 10]) * 10 ** -12 / 10 ** -6
k_variance = variance_th(k_B, T, k_values)
k_linspace_variance = variance_th(k_B, T, k_linspace)
k_linspace *= 1e6
k_linspace_variance *= 1e12

k_values_plot = k_values * 1e6
k_variance_plot = k_variance * 1e12

k_values_data = np.array([0.1, 1, 2, 5, 10]) * 10 ** -12 / 10 ** -6

displacement_different_k = np.zeros([len(k_values_data), n_timesteps_var])
for i in range(len(k_values_data)):
    displacement_different_k[i, :] = simulate_langevin(displacement_different_k[i, :], k_values_data[i], gamma, delta_t, k_B,
                                                       T, n_timesteps_var)
k_values_data_plot = k_values_data * 1e6

displacement_k = np.mean(displacement_different_k ** 2, axis=1)
displacement_k *= 1e12


fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].plot(x_vector, y_vector, 'C0', marker=',', linestyle='none', alpha=0.8)
ax[0].set_xlabel("$x$ [nm]")
ax[0].set_ylabel("$y$ [nm]")
ax[0].set_ylim([-400, 400])
ax[0].set_xlim([-400, 400])

ax[1].plot(x_bins, n_x, 'C0')
ax[1].plot(y_bins, n_y, 'green')
ax[1].plot(x_bins, p_boltzmann_x, 'k--')
ax[1].plot(y_bins, p_boltzmann_y, 'k--')
ax[1].set_xlabel("$x, y$ [nm]")
ax[1].set_ylabel("$p(x), p(y)$")
ax[1].yaxis.set_ticks([])
ax[1].set_xlim([-300, 300])

ax[2].plot(t_vector[:int(0.1 / delta_t)], x_auto[:int(0.1 / delta_t)], 'green')
ax[2].plot(t_vector[:int(0.1 / delta_t)], y_auto[:int(0.1 / delta_t)], 'C0')
ax[2].plot(t_vector, x_auto_th, 'k--')
ax[2].plot(t_vector, y_auto_th, 'k--')
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("$C(x), C(y)$")
# ax[2].yaxis.set_ticks([])
ax[2].set_xlim([0, 0.1])

# ax[3].plot(k_values_plot, k_variance_plot, 'k--')
ax[3].plot(k_linspace, k_linspace_variance, 'k--')
ax[3].plot(k_values_data_plot, displacement_k, 'ko')
ax[3].set_xlabel("$k$[pN / $\\mu m$]")
ax[3].set_ylabel("$\\langle x^2 \\rangle$ [$\\mu$m$^2$]")
ax[3].legend(["Theoretical", "Data"], loc='upper right')

plt.tight_layout()

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
h = ax2[0].hist2d(x_vector, y_vector, bins=100, density=True, cmap='rainbow')
x_v_max = max(x_vector)
x_v_min = min(x_vector)
y_v_min = min(y_vector)
y_v_max = max(y_vector)
x_boltzmann = np.linspace(x_v_min, x_v_max, 800) * 1e-9
print(x_boltzmann)
y_boltzmann = np.linspace(y_v_min, y_v_max, 800) * 1e-9
print(y_boltzmann)

ax2[0].set_xlabel("$x$ [nm]")
ax2[0].set_ylabel("$y$ [nm]")
# ax2[0].set_aspect("equal")
ax2[0].set_title("Numerical")

divider = make_axes_locatable(ax2[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
c = fig2.colorbar(h[3], ax=ax2[0], cax=cax)

# ax2[1].contour(boltzmann_2D2(x_boltzmann, y_boltzmann, k_x, k_y, k_B, T), extend=[-400, 400, -400, 400], cmap='rainbow')
# ax2[1].set_xlabel("$x$ [nm]")
# ax2[1].set_ylabel("$y$ [nm]")
# ax2[1].set_aspect("equal")

# ndim, nsamples = 2, len(x_vector)
# samples = np.array([x_vector, y_vector]).reshape([nsamples, ndim])
# figure1 = corner.corner(samples)
#
# ndim, nsamples = 2, len(p_boltzmann_x)
# samples = np.array([p_boltzmann_x, p_boltzmann_y]).reshape([nsamples, ndim])
# figure2 = corner.corner(samples)
normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

ax2[1].pcolor(x_boltzmann, y_boltzmann, boltzmann_2D(x_boltzmann, y_boltzmann, k_x, k_y, k_B, T,), cmap='rainbow', norm=normalize)
ax2[1].set_xlabel("$x$ [m]")
ax2[1].set_ylabel("$y$ [m]")
# ax2[1].set_aspect("equal")
ax2[1].set_title("Theoretical")
plt.tight_layout()

plt.show()
