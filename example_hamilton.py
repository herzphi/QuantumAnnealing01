import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


# Define Hamiltonians
def h_driver():
    # Example driver Hamiltonian (H_D)
    return np.array([[0, 1], [1, 0]])


def h_problem():
    # Example problem Hamiltonian (H_P)
    return np.array([[1, 0], [0, -1]])


# Time-dependent coefficients A(t) and B(t)
def A(t, T):
    return 1 - t / T


def B(t, T):
    return t / T


# Time evolution operator
def time_evolution_operator(H, dt):
    return expm(-1j * H * dt)


# Total Hamiltonian at time t
def H_total(t, T):
    return A(t, T) * h_driver() + B(t, T) * h_problem()


# Parameters
T = 10.0  # Total annealing time
num_steps = 1000
dt = T / num_steps

# Initial state |+> = 1/sqrt(2) (|0> + |1>)
state = np.array([1, 1]) / np.sqrt(2)

# Time evolution
ts, states = ([] for i in range(2))
for step in range(num_steps):
    t = step * dt
    H_t = H_total(t, T)
    U_t = time_evolution_operator(H_t, dt)
    state = U_t @ state  # Matrix multiplication
    ts.append(t)
    states.append(state)

states_real_0 = [np.real(el)[0] for el in states]
states_real_1 = [np.real(el)[1] for el in states]
states_imag_0 = [np.imag(el)[0] for el in states]
states_imag_1 = [np.imag(el)[1] for el in states]

fig, axs = plt.subplots(1, figsize=(5, 5))

axs.plot(ts, states_real_0, "b-", label="0 Real")
axs.plot(ts, states_real_1, "r-", label="1 Real")
axs.plot(ts, states_imag_0, "b--", label="0 Imag")
axs.plot(ts, states_imag_1, "r--", label="1 Imag")

plt.legend()
plt.show()
