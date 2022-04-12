import math

import numpy as np
import matplotlib.pyplot as plt
import cmath


class Operator:
    def __init__(self, dimension, conjugate):
        self.dimension = dimension
        self.conjugate = conjugate

    def initialize(self):
        try:
            n = self.dimension
            assert self.conjugate == 0 or self.conjugate == 1 == True
            matrix = np.zeros((n, n))
            if self.conjugate == 0:
                for i in range(n):
                    for j in range(n):
                        if i + 1 == j:
                            matrix[i][j] = np.sqrt(j)
            else:
                for i in range(n):
                    for j in range(n):
                        if j + 1 == i:
                            matrix[i][j] = np.sqrt(i)
            return matrix
        except ValueError:
            print('Please enter an integer!')
            return None
        except AssertionError:
            print('PLease enter 0 for annihilation and 1 for creation')
            return None


# initial parameters
dimension = int(input('Enter the number of dimensions: '))
omega_01 = 5
omega_R = 5
amplitude_R = 0.1
time = 100
counts = 2600
tau = time / counts
alpha = 1
beta = 0.5
sigma = 3
mu = 5
t_g = 4 * sigma
# identity matrix
I = np.zeros(dimension)
creation = Operator(dimension, 0)
creation = creation.initialize()
annihilation = Operator(dimension, 1)
annihilation = annihilation.initialize()


# initial psi (determines the initial state)
def init_psi(dimensions):
    matrix = np.zeros((dimensions, 1))
    matrix[0][0] = 1
    return matrix


psi = init_psi(dimension)

end_probability_excited = []
end_probability_ground = []
end_probability_third = []
end_Rhabi = []

for k in range(counts):
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

    oscillatory_part = amplitude_R * np.cos(omega_R * (2 * t(k + 1 / 2, tau)))
    oscillatory_part_1st_derivative = -amplitude_R * omega_R * np.sin(omega_R * (2 * t(k + 1 / 2, tau)))
    oscillatory_part_2nd_derivative = -amplitude_R * omega_R ** 2 * np.cos(omega_R * (2 * t(k + 1 / 2, tau)))

    # Crank-Nicolson 2nd order method implementation

    Hamiltonian = omega_01 * creation * annihilation - mu / 2 * creation * annihilation * (
            creation * annihilation - I) + oscillatory_part * (creation + annihilation)
    Hamiltonian_1st_derivative = oscillatory_part_1st_derivative
    Hamiltonian_2nd_derivative = oscillatory_part_2nd_derivative
    commutator = Hamiltonian_1st_derivative * Hamiltonian - Hamiltonian * Hamiltonian_1st_derivative

    F = Hamiltonian + (tau ** 2 / 24) * Hamiltonian_2nd_derivative - 1j * (tau ** 2 / 12) * commutator

    numerator = I - (tau ** 2 / 12) * np.matmul(F, F) - 1j * (tau / 2) * F
    denominator = I - (tau ** 2 / 12) * np.matmul(F, F) + 1j * (tau / 2) * F
    denominator = np.linalg.inv(denominator)
    fraction = np.matmul(denominator, numerator)

    psi = np.matmul(fraction, psi)

    Hamiltonian_0 = Hamiltonian = omega_01 * creation * annihilation - mu / 2 * creation * annihilation * (
            creation * annihilation - I) + amplitude_R * (creation + annihilation)

    eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)
    # initial excited/ground state
    excited_state = eigenpsi[0]
    ground_state = eigenpsi[1]
    if dimension >= 2:
        third_state = eigenpsi[2]
    else:
        third_state = 0
    probability_third = third_state * psi
    probability_third = abs(np.sum(probability_third * probability_third.conjugate()))
    end_probability_third.append(probability_third)
    # excited probability formula
    probability_excited = excited_state * psi
    probability_excited = abs(np.sum(probability_excited * probability_excited.conjugate()))
    end_probability_excited.append(probability_excited)
    # ground probability formula
    probability_ground = ground_state * psi
    probability_ground = abs(np.sum(probability_ground * probability_ground.conjugate()))
    end_probability_ground.append(probability_ground)

    # norm of the psi matrix
    norm = abs(np.sum(psi * psi.conjugate()))

    # Rhabi soultion for comparison
    Omega = np.sqrt((omega_R - omega_01) ** 2 + (Omega_x + Omega_y) ** 2)
    Rhabi = 1 - (Omega_x + Omega_y) ** 2 / Omega ** 2 * np.sin(Omega * t(k, tau) / 2) ** 2
    end_Rhabi.append(Rhabi)

    print(norm)

# plot image
axis = [x for x in range(counts)]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='first state')
at.plot(axis, end_probability_ground, label='second state')
at.plot(axis, end_probability_third, label='third state')
at.set_xlabel('time')
at.set_ylabel('probability')
at.set_title("States graph")
at.legend(loc='lower right')
plt.show()
