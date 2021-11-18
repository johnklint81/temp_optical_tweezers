import numpy as np
from matplotlib import pyplot as plt

n_bins = 30
n_steps = 1000
n_walks = 10000
positions_coinflip = np.zeros([n_walks, n_steps])
positions_gaussian = np.zeros([n_walks, n_steps])
positions_equal = np.zeros([n_walks, n_steps])

timesteps = np.arange(n_steps)


def evolve_coinflip(_positions_coinflip, _n_steps, _n_walks):
    _coinflip = [-1, 1]
    for i in range(_n_walks):
        for j in range(_n_steps - 1):
            _positions_coinflip[i, j + 1] = _positions_coinflip[i, j] + np.random.choice(_coinflip)
    return _positions_coinflip


def evolve_gaussian(_positions_gaussian, _n_steps, _n_walks):
    for i in range(_n_walks):
        for j in range(_n_steps - 1):
            _positions_gaussian[i, j + 1] = positions_gaussian[i, j] + np.random.normal(loc=0.0, scale=np.sqrt(1))
    return _positions_gaussian


def evolve_equal(_positions_equal, _n_steps, _n_walks):
    equal = [-1, (1 - np.sqrt(3)) / 2, (1 + np.sqrt(3)) / 2]
    for i in range(_n_walks):
        for j in range(_n_steps - 1):
            _positions_equal[i, j + 1] = _positions_equal[i, j] + np.random.choice(equal)
    return _positions_equal


positions_coinflip = evolve_coinflip(positions_coinflip, n_steps, n_walks)
positions_gaussian = evolve_gaussian(positions_gaussian, n_steps, n_walks)
positions_equal = evolve_equal(positions_equal, n_steps, n_walks)

fig, ax = plt.subplots(2, 3)
for i in range(100):
    ax[1, 0].plot(positions_coinflip[i, :1000], timesteps[:1000], color='tab:blue', alpha=0.7, linewidth=0.1)
    ax[1, 0].set_xlim([-150, 150])
    ax[1, 0].set_ylim([0, 1000])
    ax[1, 0].set_xlabel("x(t)")
    ax[1, 0].set_ylabel("t [steps]")
    ax[1, 1].plot(positions_gaussian[i, :1000], timesteps[:1000], color='tab:green', alpha=0.7, linewidth=0.1)
    ax[1, 1].set_xlim([-150, 150])
    ax[1, 1].set_ylim([0, 1000])
    ax[1, 1].set_xlabel("x(t)")
    ax[1, 1].set_ylabel("t [steps]")
    ax[1, 2].plot(positions_equal[i, :1000], timesteps[:1000], color='tab:orange', alpha=0.7, linewidth=0.1)
    ax[1, 2].set_xlim([-150, 150])
    ax[1, 2].set_ylim([0, 1000])
    ax[1, 2].set_xlabel("x(t)")
    ax[1, 2].set_ylabel("t [steps]")
    ax[0, 0].hist(positions_coinflip[:, -1], bins=n_bins, ec='white',  color='tab:blue', density=True)
    ax[0, 1].hist(positions_gaussian[:, -1], bins=n_bins, ec='white', color='tab:green', density=True)
    ax[0, 2].hist(positions_equal[:, -1], bins=n_bins, ec='white', color='tab:orange', density=True)
    ax[0, 0].set_xlim([-150, 150])
    ax[0, 1].set_xlim([-150, 150])
    ax[0, 2].set_xlim([-150, 150])
    ax[0, 0].set_box_aspect(1)
    ax[0, 1].set_box_aspect(1)
    ax[0, 2].set_box_aspect(1)
    ax[1, 0].set_box_aspect(1)
    ax[1, 1].set_box_aspect(1)
    ax[1, 2].set_box_aspect(1)

plt.tight_layout()
plt.show()
