import numpy as np
import matplotlib.pyplot as plt

sigma_z = np.array(
    [[1, 0],
     [0, -1]]
)
sigma_x = np.array(
    [[0, 1],
     [1, 0]]
)

I = np.array(
    [[1, 0],
     [0, 1]]
)
omega_01 = 5
omega_R = 5.1
amplitude_R = 0.01
time = 10
counts = 100
tau = time / counts

psi = np.array(
    [1, 0]
)
end_probability_excited = []
end_probability_ground = []

for k in range(counts):
    excited_state = np.array(
        [1, 0]
    )
    ground_state = np.array(
        [0, 1]
    )
    numerator = I - (tau / 2) * (omega_01 * sigma_z + amplitude_R * np.cos(omega_R * (k * tau + tau / 2)) * sigma_x)
    denominator = I + (tau / 2) * (omega_01 * sigma_z + amplitude_R * np.cos(omega_R * (k * tau + tau / 2)) * sigma_x)
    denominator = np.linalg.inv(denominator)
    fraction = np.matmul(numerator, denominator)
    psi = np.matmul(fraction, psi)
    probability_excited = np.matmul(excited_state, psi)
    probability_excited = probability_excited ** 2
    end_probability_excited.append(probability_excited)
    probability_ground = np.matmul(ground_state, psi)
    probability_ground = probability_ground ** 2
    end_probability_ground.append(probability_ground)
    print(psi)
axis = [x for x in range(counts)]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='excited state')
at.plot(axis, end_probability_ground, label='ground state')
at.set_xlabel('x label')
at.set_ylabel('y label')
at.set_title("States graph")
at.legend()
plt.show()
