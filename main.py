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


def alpha_state(dim, n):
    try:
        assert n in range(1, 7, 1)
        alpha_1 = np.zeros((dim, 1))
        alpha_1[1][0] = 1
        alpha_2 = np.zeros((dim, 1))
        alpha_2[0][0] = 1
        if n == 1:
            return alpha_1
        elif n == 2:
            return alpha_2
        elif n == 3:
            return 1 / np.sqrt(2) * (alpha_1 + alpha_2)
        elif n == 4:
            return 1 / np.sqrt(2) * (alpha_1 - alpha_2)
        elif n == 5:
            return 1 / np.sqrt(2) * (alpha_1 + alpha_2 * 1j)
        elif n == 6:
            return 1 / np.sqrt(2) * (alpha_1 - alpha_2 * 1j)
        else:
            return None
    except ValueError:
        print('Please enter an integer!')
    except AssertionError:
        print('PLease enter numbers from 1 to 6')
        return None


# initial parameters
try:
    rotation_type = str(input('Enter the rotation axis:'))
    assert rotation_type in ['x', 'y']
except ValueError:
    print('Wrong value type')
    quit()
except AssertionError:
    print('Please, enter x or y')
    quit()

phase = 0 if rotation_type == 'y' else np.pi / 2
dimension = int(input('Enter the number of dimensions: '))
omega_01 = 5
omega_R = 5
amplitude_R = 0.1
time = 30
counts = 3000
tau = time / counts
alpha = 1
beta = 0.5
sigma = 3
mu = 0.3
t_g = 4 * sigma
# identity matrix
I = np.identity(dimension)
annihilation = Operator(dimension, 0)
annihilation = annihilation.initialize()
creation = Operator(dimension, 1)
creation = creation.initialize()


# initial psi (determines the initial state)
def init_psi(dimensions):
    matrix = np.zeros((dimensions, 1))
    matrix[0][0] = 1
    return matrix


def Rhabi_solution():
    for k in range(counts):
        # Start of psi calculation
        t = lambda k_: k_ * tau
        # Oscillatory amplitude (related to DRAG techinque)
        Omega_x_numerator = np.exp(-(t(k) - 0.5 * t_g) ** 2 / (2 * sigma ** 2)) - np.exp(
            -t_g ** 2 / (8 * sigma ** 2))
        Omega_x_denominatior = np.sqrt(2 * np.pi * sigma ** 2) * math.erf(t_g / (np.sqrt(8) * sigma)) - t_g * np.exp(
            -t_g ** 2 / (8 * sigma ** 2))
        Omega_x = alpha * amplitude_R * Omega_x_numerator / Omega_x_denominatior

        Omega_y_numerator = np.exp(-(t(k) - 0.5 * t_g) ** 2 / (2 * sigma ** 2)) * (t(k) - 0.5 * t_g)
        Omega_y_denominator = (sigma ** 2) * Omega_x_denominatior
        Omega_y = -beta * amplitude_R * (-Omega_y_numerator / Omega_y_denominator)

        # Rhabi soultion for comparison
        Omega = np.sqrt((omega_R - omega_01) ** 2 + (Omega_x + Omega_y) ** 2)
        Rhabi = 1 - (Omega_x + Omega_y) ** 2 / Omega ** 2 * np.sin(Omega * t(k) / 2) ** 2
        end_Rhabi.append(Rhabi)


end_probability_excited = []
end_probability_ground = []
end_probability_third = []
end_Rhabi = []

# Finding eigenvalues and eigenvectors
Hamiltonian_0 = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
        np.matmul(creation, annihilation) - I)
eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)

# Main cycle

psi = init_psi(dimension)
for k in range(counts):
    # Start of psi calculation
    t = lambda k_: k_ * tau

    oscillatory_part = amplitude_R * np.cos(omega_R * (t(k + 1 / 2)) + phase)
    oscillatory_part_1st_derivative = -amplitude_R * omega_R * np.sin(omega_R * (t(k + 1 / 2)) + phase)
    oscillatory_part_2nd_derivative = -amplitude_R * omega_R ** 2 * np.cos(omega_R * (t(k + 1 / 2)) + phase)

    # Crank-Nicolson 2nd order method implementation

    Hamiltonian = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
            np.matmul(creation, annihilation) - I) + oscillatory_part * (creation + annihilation)
    Hamiltonian_1st_derivative = oscillatory_part_1st_derivative
    Hamiltonian_2nd_derivative = oscillatory_part_2nd_derivative
    commutator = Hamiltonian_1st_derivative * Hamiltonian - Hamiltonian * Hamiltonian_1st_derivative

    F = Hamiltonian + (tau ** 2 / 24) * Hamiltonian_2nd_derivative - 1j * (tau ** 2 / 12) * commutator

    numerator = I - (tau ** 2 / 12) * np.matmul(F, F) - 1j * (tau / 2) * F
    denominator = I - (tau ** 2 / 12) * np.matmul(F, F) + 1j * (tau / 2) * F
    denominator = np.linalg.inv(denominator)
    fraction = np.matmul(denominator, numerator)

    psi = np.matmul(fraction, psi)

    # initial excited/ground state
    excited_state = np.array(eigenpsi[:, [1]])
    ground_state = np.array(eigenpsi[:, [0]])
    if dimension >= 3:
        third_state = np.array(eigenpsi[:, [2]])

        # third state probability formula
        probability_third = np.matmul(third_state.transpose(), psi.conjugate())
        probability_third = abs(np.sum(probability_third)) ** 2
        end_probability_third.append(probability_third)

    else:
        third_state = 0
        probability_third = 0
        end_probability_third.append(probability_third)

    # excited probability formula
    probability_excited = np.matmul(excited_state.transpose(), psi.conjugate())
    probability_excited = abs(np.sum(probability_excited)) ** 2
    end_probability_excited.append(probability_excited)

    # ground probability formula
    probability_ground = np.matmul(ground_state.transpose(), psi.conjugate())
    probability_ground = abs(np.sum(probability_ground)) ** 2
    end_probability_ground.append(probability_ground)

    # leakage
    leakage = probability_third
    # norm of the psi matrix
    norm = abs(np.sum(psi * psi.conjugate()))

# Fidelity calculation
rotation_core = np.array([[0, -1j], [1, 0]]) if rotation_type == 'y' else np.array([[0, 1], [-1j, 0]])
rotation_matrix = np.identity(dimension, dtype='cfloat')
rotation_matrix[0:2, 0:2] = rotation_core
fidelity = 0


def wave_function_calculation(wave_function):
    for k in range(counts):
        # Start of psi calculation
        t = lambda k_: k_ * tau
        oscillatory_part = amplitude_R * np.cos(omega_R * (t(k + 1 / 2)) + phase)
        oscillatory_part_1st_derivative = -amplitude_R * omega_R * np.sin(omega_R * (t(k + 1 / 2)) + phase)
        oscillatory_part_2nd_derivative = -amplitude_R * omega_R ** 2 * np.cos(omega_R * (t(k + 1 / 2)) + phase)

        # Crank-Nicolson 2nd order method implementation
        Hamiltonian = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
                np.matmul(creation, annihilation) - I) + oscillatory_part * (creation + annihilation)
        Hamiltonian_1st_derivative = oscillatory_part_1st_derivative
        Hamiltonian_2nd_derivative = oscillatory_part_2nd_derivative
        commutator = Hamiltonian_1st_derivative * Hamiltonian - Hamiltonian * Hamiltonian_1st_derivative

        F = Hamiltonian + (tau ** 2 / 24) * Hamiltonian_2nd_derivative - 1j * (tau ** 2 / 12) * commutator

        numerator = I - (tau ** 2 / 12) * np.matmul(F, F) - 1j * (tau / 2) * F
        denominator = I - (tau ** 2 / 12) * np.matmul(F, F) + 1j * (tau / 2) * F
        denominator = np.linalg.inv(denominator)

        fraction = np.matmul(denominator, numerator)
        wave_function = np.matmul(fraction, wave_function)
    return wave_function


for i in range(1, 7, 1):
    psi_g = np.matmul(rotation_matrix, alpha_state(dimension, i))
    probability = np.matmul(wave_function_calculation(alpha_state(dimension, i)).transpose(), psi_g.conjugate())
    probability = abs(np.sum(probability)) ** 2
    fidelity += 1 / 6 * probability
    print(fidelity)

print('The fidelity is:', str(fidelity), sep=' ')
print('The leakage is:', str(leakage))

# plot image
axis = [tau * x for x in range(counts)]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='excited state')
at.plot(axis, end_probability_ground, label='ground state')
at.plot(axis, end_probability_third, label='third state')
# at.plot(axis, end_Rhabi, label='Rhabi')
at.set_xlabel('time, ms')
at.set_ylabel('probability')
at.set_title("States graph")
at.legend(loc='lower right')
plt.show()
