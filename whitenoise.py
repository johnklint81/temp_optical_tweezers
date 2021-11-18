import numpy as np
from matplotlib import pyplot as plt

delta_t1 = 0.01
delta_t2 = 0.05
delta_t3 = 0.1

n_timesteps_1 = int(5 / delta_t1)
n_timesteps_2 = int(5 / delta_t2)
n_timesteps_3 = int(5 / delta_t3)

n_simulations = 50
n_simulations_long = int(1e4)

t_vector_1 = np.linspace(0, 5, n_timesteps_1)
t_vector_2 = np.linspace(0, 5, n_timesteps_2)
t_vector_3 = np.linspace(0, 5, n_timesteps_3)


def simulate_noise(_n_timesteps, delta_t):
    _x = np.zeros(_n_timesteps)
    for i in range(_n_timesteps - 1):
        r = np.random.normal(loc=0, scale=1)
        _x[i + 1] = _x[i] + r * np.sqrt(delta_t)
    return _x


x_1 = np.zeros([n_simulations, n_timesteps_1])
for i in range(n_simulations):
    x_1[i, :] = simulate_noise(n_timesteps_1, delta_t1)

x_2 = np.zeros([n_simulations, n_timesteps_2])
for i in range(n_simulations):
    x_2[i, :] = simulate_noise(n_timesteps_2, delta_t2)

x_3 = np.zeros([n_simulations, n_timesteps_3])
for i in range(n_simulations):
    x_3[i, :] = simulate_noise(n_timesteps_3, delta_t3)


x_1_long = np.zeros([n_simulations_long, n_timesteps_1])
for i in range(n_simulations_long):
    x_1_long[i, :] = simulate_noise(n_timesteps_1, delta_t1)

x_2_long = np.zeros([n_simulations_long, n_timesteps_2])
for i in range(n_simulations_long):
    x_2_long[i, :] = simulate_noise(n_timesteps_2, delta_t2)

x_3_long = np.zeros([n_simulations_long, n_timesteps_3])
for i in range(n_simulations_long):
    x_3_long[i, :] = simulate_noise(n_timesteps_3, delta_t3)


MSD_vector_1 = x_1_long ** 2
mean_MSD_vector_1 = np.mean(MSD_vector_1, axis=0)
MSD_vector_2 = x_2_long ** 2
mean_MSD_vector_2 = np.mean(MSD_vector_2, axis=0)
MSD_vector_3 = x_3_long ** 2
mean_MSD_vector_3 = np.mean(MSD_vector_3, axis=0)

MSD_1 = np.mean(mean_MSD_vector_1[-1]) / 5
MSD_2 = np.mean(mean_MSD_vector_2[-1]) / 5
MSD_3 = np.mean(mean_MSD_vector_3[-1]) / 5

print(f'\nMSD1: {MSD_1}\nMSD2: {MSD_2}\nMSD3: {MSD_3}')

fig, ax = plt.subplots(2, 3)

for i in range(n_simulations):
    ax[0, 0].plot(t_vector_1, x_1[i, :], color='tab:blue', alpha=0.9, linewidth=0.1)
    ax[0, 0].set_box_aspect(1)
    ax[0, 0].set_xlim([0, 5])
    ax[0, 0].set_ylim([-8, 8])
    ax[0, 0].set_ylabel("$x(t)$")
    ax[0, 0].set_xlabel("$t$ [s]")
    ax[0, 0].set_title(f"$\Delta t = {delta_t1}$")

    ax[0, 1].plot(t_vector_2, x_2[i, :], color='tab:green', alpha=0.9, linewidth=0.1)
    ax[0, 1].set_box_aspect(1)
    ax[0, 1].set_xlim([0, 5])
    ax[0, 1].set_ylim([-8, 8])
    ax[0, 1].set_ylabel("$x(t)$")
    ax[0, 1].set_xlabel("$t$ [s]")
    ax[0, 1].set_title(f"$\Delta t = {delta_t2}$")

    ax[0, 2].plot(t_vector_3, x_3[i, :], color='tab:orange', alpha=0.9, linewidth=0.1)
    ax[0, 2].set_box_aspect(1)
    ax[0, 2].set_xlim([0, 5])
    ax[0, 2].set_ylim([-8, 8])
    ax[0, 2].set_ylabel("$x(t})$")
    ax[0, 2].set_xlabel("$t$ [s]")
    ax[0, 2].set_title(f"$\Delta t = {delta_t3}$")

for i in range(n_simulations_long):
    ax[1, 0].plot(t_vector_1, mean_MSD_vector_1, color='tab:blue', alpha=0.7, linewidth=4)
    ax[1, 0].plot(t_vector_1, MSD_1 * t_vector_1, color='black', linestyle='dashed')
    ax[1, 0].set_box_aspect(1)
    ax[1, 0].set_xlim([0, 5])
    ax[1, 0].set_ylim([0, 6])
    ax[1, 0].set_ylabel("$\\langle x(t)^2\\rangle$")
    ax[1, 0].set_xlabel("$t$ [s]")

    ax[1, 1].plot(t_vector_2, mean_MSD_vector_2, color='tab:green', alpha=0.7, linewidth=4)
    ax[1, 1].plot(t_vector_2, MSD_2 * t_vector_2, color='black', linestyle='dashed')
    ax[1, 1].set_box_aspect(1)
    ax[1, 1].set_xlim([0, 5])
    ax[1, 1].set_ylim([0, 6])
    ax[1, 1].set_ylabel("$\\langle x(t)^2\\rangle$")
    ax[1, 1].set_xlabel("$t$ [s]")

    ax[1, 2].plot(t_vector_3, mean_MSD_vector_3, color='tab:orange', alpha=0.7, linewidth=4)
    ax[1, 2].plot(t_vector_3, MSD_3 * t_vector_3, color='black', linestyle='dashed')
    ax[1, 2].set_box_aspect(1)
    ax[1, 2].set_xlim([0, 5])
    ax[1, 2].set_ylim([0, 6])
    ax[1, 2].set_ylabel("$\\langle x(t)^2\\rangle$")
    ax[1, 2].set_xlabel("$t$ [s]")

plt.tight_layout()
plt.show()
