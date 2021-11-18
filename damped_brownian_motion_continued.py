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
long_time = 1e6
n_timesteps = int(100 * tau / delta_t)
n_timesteps_long = int(long_time * tau / delta_t)
t_vector = np.linspace(0, 100, n_timesteps)
t_vector_long = np.linspace(0, long_time, n_timesteps_long)
parameters = [R, m, eta, gamma, k_B, T, delta_t, n_timesteps]
parameters_long = [R, m, eta, gamma, k_B, T, delta_t, n_timesteps_long]
n_realizations = 100


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


x_1_long, x_2_long = simulate_langevin(t_vector_long, parameters_long)
x_1_long_squared = x_1_long ** 2
x_1_long_squared_cumsum = np.cumsum(x_1_long_squared)
x_2_long_squared = x_2_long ** 2
x_2_long_squared_cumsum = np.cumsum(x_2_long_squared)
MSD_1_long = np.divide(x_1_long_squared_cumsum, np.arange(0, n_timesteps_long, 1))
MSD_2_long = np.divide(x_2_long_squared_cumsum, np.arange(0, n_timesteps_long, 1))

x_1_many = np.zeros([n_realizations, n_timesteps])
x_2_many = np.zeros_like(x_1_many)

for i in range(n_realizations):
    x_1_many[i, :], x_2_many[i, :] = simulate_langevin(t_vector, parameters)

MSD_1_array = x_1_many ** 2
MSD_1 = np.mean(MSD_1_array, axis=0)
MSD_2_array = x_2_many ** 2
MSD_2 = np.mean(MSD_2_array, axis=0)

MSD_1_many_k = (MSD_1[-2] - MSD_1[int(n_timesteps / 1.5)]) / (t_vector[-2] - t_vector[int(n_timesteps / 1.5)]) * 1e18
MSD_2_many_k = (MSD_2[-2] - MSD_2[int(n_timesteps / 1.5)]) / (t_vector[-2] - t_vector[int(n_timesteps / 1.5)]) * 1e18

print(f'Inertial k-value: {MSD_1_many_k}')
print(f'Non-inertial k-value: {MSD_2_many_k}')

fig, ax = plt.subplots(figsize=(16, 8))

ax.set_box_aspect(1)
ax.loglog()
ax.plot(t_vector_long[1:-1], MSD_1_long[1:-1] * 1e18, 'b--', linewidth='3', alpha=0.8)
ax.plot(t_vector_long[1:-1], MSD_2_long[1:-1] * 1e18, 'k--')
ax.plot(t_vector_long[1:-1], MSD_1_many_k * t_vector_long[1:-1], 'g', alpha=0.8)
ax.plot(t_vector_long[1:-1], MSD_2_many_k * t_vector_long[1:-1], 'r-.', alpha=0.8)
ax.set_ylim([1e-4, long_time])
ax.set_xlim([0.1, long_time])
ax.set_ylabel("$\\langle x^2 \\rangle$ [nm$^2$]")
ax.set_xlabel("$t/\\tau$")
ax.legend(["Inertial: time average", "Non-inertial: time average", "Inertial: ensemble average",
           "Non-inertial: ensemble average"], loc='lower right')

plt.show()
