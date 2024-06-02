import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, expm_multiply

# Constants and parameters
N = 500  # Number of spatial points
L = 10.0  # Length of the spatial domain
dx = L / N  # Spatial step size
T = 10.0  # Total annealing time
dt = 0.01  # Time step size
num_steps = int(T / dt)

# Spatial grid
x = np.linspace(-L / 2, L / 2, N)


# Double-well potential
def double_well_potential(x):
    return x**4 - 10 * x**2


V = double_well_potential(x)

# Hamiltonian matrices
H_P = diags([V], [0]).toarray()
H_D = -0.5 * diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray() / dx**2

# Get the ground state of the driver Hamiltonian
_, psi0 = eigsh(H_D, k=1, which="SA")
psi = psi0.flatten()


# Schedule function
def schedule(t):
    return t / T


# Time evolution using expm_multiply for better stability
for t in range(num_steps):
    s = schedule(t * dt)
    H_t = (1 - s) * H_D + s * H_P
    psi = expm_multiply(-1j * H_t * dt, psi)
    psi /= np.linalg.norm(psi)  # Normalize the wave function

# Calculate the probability density
prob_density = np.abs(psi) ** 2

# Plotting
import matplotlib.pyplot as plt

plt.plot(x, prob_density, label="Probability Density")
plt.plot(
    x,
    double_well_potential(x) / max(double_well_potential(x)),
    label="Potential (normalized)",
)
plt.vlines(np.sqrt(5), -0.1, 0.1, "black", "--", r"$V(x)_{min}$")
plt.vlines(-np.sqrt(5), -0.1, 0.1, "black", "--")

plt.xlim(-4, 4)
plt.ylim(-0.1, 0.1)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/onedim.png", format="png", dpi=300)
