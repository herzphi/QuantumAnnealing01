import numpy as np
from scipy.sparse import diags
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Constants and parameters
N = 200  # Number of spatial points
L = 10.0  # Length of the spatial domain
dx = L / N  # Spatial step size
T = 10.0  # Total annealing time
dt = 0.01  # Time step size
num_steps = int(T / dt)

# Spatial grid
x = np.linspace(-L / 2, L / 2, N)


# Double-well potential
def double_well_potential(x):
    return x**4 - x**2


V = double_well_potential(x)

# Hamiltonian matrices
H_P = diags([V], [0]).toarray()
H_D = -0.5 * diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray() / dx**2

# Initial wave function (ground state of H_D)
psi = np.exp(-(x**2))
psi /= np.linalg.norm(psi)


# Schedule function
def schedule(t):
    return t / T


# Time evolution operator (Crank-Nicolson method)
def time_evolution_operator(H, dt):
    return expm(-1j * H * dt)


# Time evolution
for t in range(num_steps):
    s = schedule(t * dt)
    H_t = (1 - s) * H_D + s * H_P
    U = time_evolution_operator(H_t, dt)
    psi = U @ psi
    psi /= np.linalg.norm(psi)  # Normalize the wave function

# Calculate the probability density
prob_density = np.abs(psi) ** 2

# Plotting

plt.plot(x, prob_density, label="Probability Density")
plt.plot(
    x,
    double_well_potential(x) / max(double_well_potential(x)),
    label="Potential (normalized)",
)
plt.vlines(-1 / np.sqrt(2), -0.01, 0.04)
plt.vlines(1 / np.sqrt(2), -0.01, 0.04)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.ylim(-0.01, 0.04)
plt.legend()
plt.savefig("./plots/onedim.png", format="png", dpi=300)
