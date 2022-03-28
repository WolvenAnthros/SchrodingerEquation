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
omega_R = 5.4
amplitude_R = 2
time = 1.5
counts = 111
tau = time / counts
# initial psi (determines the initial state)
psi = np.array(
    [1, 0]
    , dtype=complex)

end_probability_excited = []
end_probability_ground = []

for k in range(counts):
    oscillatory_part = amplitude_R * np.cos(omega_R * (k * tau + tau / 2))
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
    # Crank-Nicolson method implementation
    numerator = I - 1j * (tau / 2) * (
            omega_01 * sigma_z + oscillatory_part * sigma_x)
    denominator = I + 1j * (tau / 2) * (
            omega_01 * sigma_z + oscillatory_part * sigma_x)
    denominator = np.linalg.inv(denominator)
    fraction = np.matmul(denominator, numerator)
    psi = np.matmul(fraction, psi)
    # norm of the psi matrix
    norm = abs(np.matmul(psi, psi.conjugate()))
    print(norm)

# plot image
axis = [x for x in range(counts)]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='excited state')
at.plot(axis, end_probability_ground, label='ground state')
at.set_xlabel('time')
at.set_ylabel('probability')
at.set_title("States graph")
at.legend()
plt.show()
