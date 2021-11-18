import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

R = 1e-6
m = 1.11 * 1e-14
eta = 0.001
gamma = 6 * np.pi * eta * R
T = 300
k_B = 1.38 * 1e-23

tau = m / gamma
delta_t = 0.01 * tau
long_time = 1e5
n_timesteps = int(100 * tau / delta_t)
t_vector = np.linspace(0, 100, n_timesteps)
parameters = [R, m, eta, gamma, k_B, T, delta_t, n_timesteps]
n_realizations = 10000


def simulate_langevin(_t_vector, _parameters):
    _x_1 = np.zeros_like(_t_vector)
    _x_2 = np.zeros_like(_t_vector)
    _n_timesteps = _parameters[-1]
    _delta_t = _parameters[-2]
    _gamma = _parameters[3]
    _m = _parameters[1]
    _k_B = _parameters[4]
    _T = _parameters[5]
    for i in range(_n_timesteps - 2):
        _w = np.random.normal(loc=0, scale=1)

        # Undamped
        _x_1[i + 2] = (2 + _delta_t * (_gamma / _m)) / (1 + _delta_t * (_gamma / _m)) * _x_1[i + 1] - \
                    1 / (1 + _delta_t * (_gamma / _m)) * _x_1[i] + \
                    np.sqrt(2 * k_B * _T * _gamma) / (_m + _delta_t * _gamma) * _delta_t ** (3 / 2) * _w

        # Damped
        _x_2[i + 1] = _x_2[i] + np.sqrt(2 * _k_B * _T * _delta_t / _gamma) * _w

    return _x_1, _x_2


x_1 = np.zeros([n_realizations, n_timesteps])
x_2 = np.zeros_like(x_1)

for i in range(n_realizations):
    x_1[i, :], x_2[i, :] = simulate_langevin(t_vector, parameters)


duration = int(tau/delta_t)
max_x2 = max(abs(x_2[0, :duration])) * 1e9

MSD_1_array = x_1 ** 2
MSD_1 = np.mean(MSD_1_array, axis=0)
MSD_2_array = x_2 ** 2
MSD_2 = np.mean(MSD_2_array, axis=0)

MSD_1_plot = np.geomspace(1, n_timesteps, num=60, dtype=int)
MSD_1 = [MSD_1[i - 1] for i in MSD_1_plot]
MSD_1 = np.array(MSD_1) * 1e18

t_vector_geom = [t_vector[i - 1] for i in MSD_1_plot]

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

ax[0].plot(t_vector[:duration], x_2[0, :duration] * 1e9, 'C0', alpha=0.8)
ax[0].plot(t_vector[:duration], x_1[0, :duration] * 1e9, 'k--', alpha=1)
ax[0].set_box_aspect(1)
ax[0].set_xlim([0, 1])
ax[0].set_ylim([-max_x2, max_x2])
ax[0].set_ylabel("$x$ [nm]")
ax[0].set_xlabel("$t/\\tau$")

duration = int(100*tau/delta_t)
max_x2 = max(abs(x_2[0, :])) * 1e9

ax[1].set_box_aspect(1)
ax[1].plot(t_vector[:duration], x_2[0, :duration] * 1e9, 'C0', alpha=0.8)
ax[1].plot(t_vector[:duration], x_1[0, :duration] * 1e9, 'k--', alpha=1)
ax[1].set_xlim([0, 100])
ax[1].set_ylim([-max_x2, max_x2])
ax[1].set_ylabel("$x$ [nm]")
ax[1].set_xlabel("$t/\\tau$")

ax[2].set_box_aspect(1)
ax[2].loglog()
ax[2].plot(t_vector_geom, MSD_1, 'C0', marker='o', linestyle='none', alpha=0.8, label="Inertial")
ax[2].plot(t_vector[:-2], MSD_2[:-2] * 1e18, 'k--', label="Non-inertial")
ax[2].set_ylim([0.01, 100])
ax[2].set_xlim([0.1, 100])
ax[2].set_ylabel("$\\langle x^2 \\rangle$ [nm$^2$]")
ax[2].set_xlabel("$t/\\tau$")
ax[2].legend(["Inertial", "Non-inertial"], loc='lower right')

plt.tight_layout()
plt.show()
