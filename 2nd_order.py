import numpy as np
import matplotlib.pyplot as plt
import cmath

# Pauli matrices
sigma_z = np.array(
    [[1, 0],
     [0, -1]]
)
sigma_x = np.array(
    [[0, 1],
     [1, 0]]
)
# identity matrix
I = np.array(
    [[1, 0],
     [0, 1]]
)
# initial parameters
omega_01 = 5
omega_R = 5
amplitude_R = 0.1
time = 100
counts = 2600
tau = time / counts
# initial psi (determines the initial state)
psi = np.array(
    [1, 0]
    , dtype=complex)

end_probability_excited = []
end_probability_ground = []
end_Rhabi = []
for k in range(counts):
    t_k = lambda k, tau: k * tau
    oscillatory_part = amplitude_R * np.cos(omega_R * (2 * t_k(k + 1 / 2, tau)))
    # initial excited/ground state
    excited_state = np.array(
        [1, 0]
    )
    ground_state = np.array(
        [0, 1]
    )
    # excited probability formula
    probability_excited = np.matmul(excited_state, psi)
    probability_excited = abs(probability_excited * probability_excited.conjugate())
    end_probability_excited.append(probability_excited)
    # ground probability formula
    probability_ground = np.matmul(ground_state, psi)
    probability_ground = abs(probability_ground * probability_ground.conjugate())
    end_probability_ground.append(probability_ground)
    # Crank-Nicolson 2nd order method implementation
    Hamiltonian = omega_01 * sigma_z + oscillatory_part * sigma_x
    Hamiltonian_1st_derivative = -amplitude_R * omega_R * np.sin(omega_R * (t_k(k + 1 / 2, tau)))
    Hamiltonian_2nd_derivative = -amplitude_R * (omega_R ** 2) * np.cos(omega_R * (t_k(k + 1 / 2, tau)))
    commutator = Hamiltonian_1st_derivative * Hamiltonian - Hamiltonian * Hamiltonian_1st_derivative
    F = Hamiltonian + (tau ** 2 / 24) * Hamiltonian_2nd_derivative - 1j * (tau ** 2 / 12) * commutator
    numerator = I - (tau ** 2 / 12) * np.matmul(F, F) - 1j * (tau / 2) * F
    denominator = I - (tau ** 2 / 12) * np.matmul(F, F) + 1j * (tau / 2) * F
    denominator = np.linalg.inv(denominator)
    fraction = np.matmul(denominator, numerator)
    psi = np.matmul(fraction, psi)
    # norm of the psi matrix
    norm = abs(np.matmul(psi, psi.conjugate()))
    # Rhabi soultion for comparison
    Omega = np.sqrt((omega_R - omega_01) ** 2 + amplitude_R ** 2)
    Rhabi = 1 - amplitude_R ** 2 / Omega**2 * np.sin(Omega * t_k(k, tau) / 2) ** 2
    end_Rhabi.append(Rhabi)
    print(norm)

# plot image
axis = [x for x in range(counts)]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='excited state')
at.plot(axis, end_probability_ground, label='ground state')
at.plot(axis, end_Rhabi, label='Rhabi solution')
at.set_xlabel('time')
at.set_ylabel('probability')
at.set_title("States graph")
at.legend()
plt.show()
