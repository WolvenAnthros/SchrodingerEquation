import math

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
sigma_y = np.array(
    [[0, -1j],
     [1j, 0]]
)
# identity matrix
I = np.array(
    [[1, 0],
     [0, 1]]
)
# initial parameters
omega_01 = 5
omega_R = 5
amplitude_R = np.pi
time = 100
counts = 2600
tau = time / counts
alpha = 1
beta = 0.5
sigma = 3
t_g = 4 * sigma
# initial psi (determines the initial state)
psi = np.array(
    [1, 0]
    , dtype=complex)

end_probability_excited = []
end_probability_ground = []
end_Rhabi = []

for k in range(counts):
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
    # Start of psi calculation
    t = lambda k_, tau_: k_ * tau_
    # Oscillatory amplitude (related to DRAG techinque)
    Omega_x_numerator = np.exp(-(t(k, tau) - 0.5 + t_g) ** 2 / (2 * sigma ** 2)) - np.exp(
        -t_g ** 2 / (8 * sigma ** 2))
    Omega_x_denominatior = np.sqrt(2 * np.pi * sigma ** 2) * math.erf(t_g / (np.sqrt(8) * sigma)) - t_g * np.exp(
        -t_g ** 2 / (8 * sigma ** 2))
    Omega_x = alpha * amplitude_R * Omega_x_numerator / Omega_x_denominatior

    Omega_y_numerator = np.exp(-(t(k, tau) - 0.5 + t_g) ** 2 / (2 * sigma ** 2)) * (t(k, tau) - 0.5 * t_g)
    Omega_y_denominator = (sigma ** 2) * Omega_x_denominatior
    Omega_y = -beta * amplitude_R * (-Omega_y_numerator / Omega_y_denominator)

    oscillatory_part = Omega_x * np.cos(omega_R * (2 * t(k + 1 / 2, tau))) + Omega_y * np.sin(
        omega_R * (2 * t(k + 1 / 2, tau)))
    oscillatory_part_1st_derivative = -Omega_x * omega_R * np.sin(
        omega_R * (t(k + 1 / 2, tau))) + Omega_y * omega_R * np.cos(omega_R * (2 * t(k + 1 / 2, tau)))
    oscillatory_part_2nd_derivative = -Omega_x * omega_R ** 2 * np.cos(
        omega_R * (t(k + 1 / 2, tau))) - Omega_y * omega_R ** 2 * np.sin(omega_R * (2 * t(k + 1 / 2, tau)))

    # Crank-Nicolson 2nd order method implementation
    # ////// Should these matrices (sigma_x, sigma_y) be applied to Hamiltonian in the new theory? ///////
    Hamiltonian = omega_01 * sigma_z + oscillatory_part # * (sigma_x + sigma_y)
    Hamiltonian_1st_derivative = oscillatory_part_1st_derivative # * (sigma_x + sigma_y)
    Hamiltonian_2nd_derivative = oscillatory_part_2nd_derivative # * (sigma_x + sigma_y)
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
    Omega = np.sqrt((omega_R - omega_01) ** 2 + (Omega_x + Omega_y) ** 2)
    Rhabi = 1 - (Omega_x + Omega_y) ** 2 / Omega ** 2 * np.sin(Omega * t(k, tau) / 2) ** 2
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
at.legend(loc='lower right')
plt.show()
