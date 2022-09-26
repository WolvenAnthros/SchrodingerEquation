import math
import numpy as np
import matplotlib.pyplot as plt


class Operator:
    def __init__(self, dimension, conjugate):
        self.dimension = dimension
        self.conjugate = conjugate
        self.matrix = np.zeros((self.dimension, self.dimension))
        try:
            n = self.dimension
            assert self.conjugate == 0 or self.conjugate == 1 == True
            if self.conjugate == 0:
                for i in range(n):
                    for j in range(n):
                        if i + 1 == j:
                            self.matrix[i][j] = np.sqrt(j)
            else:
                for i in range(n):
                    for j in range(n):
                        if j + 1 == i:
                            self.matrix[i][j] = np.sqrt(i)
        except ValueError:
            print('Please enter an integer!')
            self.matrix = None
        except AssertionError:
            print('PLease enter 0 for annihilation and 1 for creation')
            self.matrix = None

    def __repr__(self):
        return f'{self.matrix}'


class Psi:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.matrix = np.zeros((self.dimensions, 1))
        self.matrix[0][0] = 1


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


'''
Initial params
'''
phase = 0
dimension = 4
omega_01 = 5 * 2 * np.pi
omega_osc = 25 * 2 * np.pi
pulse_period = 2 * np.pi / omega_osc
amp = 3
time = 200
pulse_time = 0.004
mu = 0.25 * 2 * np.pi
#pulses = [1 for x in range(120)]

'''
Matrices section
'''
I = np.identity(dimension) #Identity matrix
annihilation = Operator(dimension, 0).matrix #annihilation/creation operators
creation = Operator(dimension, 1).matrix
psi = Psi(dimension).matrix # create psi matrix

# Finding eigenvalues and eigenvectors
Hamiltonian_0 = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
        np.matmul(creation, annihilation) - I)
eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)

# Irregular pulse array example
pulse_list = [0, 1, 1, 1, -1, 0, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0,
              1, 1, -1, -1, 0, 1, 1, -1, -1,
              0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0,
              1, 1, -1, -1, 0, 1, 1, -1, -1,
              0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 0, 1, 1, 1, -1, 0, 1, 1, -1, -1, 0, 1, 1, -1, -1, 1, 1, -1, -1, -1, 0,
              1, 1, -1, 1, 0, 1, -1, -1, -1, 0,
              1, -1, -1, -1]
#pulse_list = [1 for x in range(30)]
# pulse_positive = [pulse for pulse in pulse_list[::2]]
# pulse_negative = [pulse for pulse in pulse_list[1::2]]


for n in ['excited', 'ground', 'third']:
    globals()['end_probability_%s' % n] = []  # создать пустые списки end_probability_excited|ground|third

for pulse,index in zip(pulse_list,range(len(pulse_list))):
    '''
    Please pay attention that the pulse_period and pulse_time should converge well, otherwise
    we lose some of the counts
    '''
    counts = int(pulse_period * 2500)  # let's take a lot of counts for more reliability

    tau = pulse_period / counts # 0.0004
    print(f'Pulse:{pulse}, index: {index}')
    for k in range(counts):
        # Start of psi calculation
        t = lambda k_: k_ * tau
        '''
        Defining pulse shape in time
        '''
        if t(k) < pulse_time:
            oscillatory_part = amp * pulse
            oscillatory_part_1st_derivative = 0
            oscillatory_part_2nd_derivative = 0
        # elif pulse_period / 2 < t(k) < (pulse_period / 2 + pulse_time):
        #     oscillatory_part = -amp * pulse_neg
        #     oscillatory_part_1st_derivative = 0
        #     oscillatory_part_2nd_derivative = 0
        else:
            oscillatory_part = 0
            oscillatory_part_1st_derivative = 0
            oscillatory_part_2nd_derivative = 0

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

        # excited probability formula
        probability_excited = np.matmul(excited_state.transpose(), psi.conjugate())
        probability_excited = abs(np.sum(probability_excited)) ** 2
        end_probability_excited.append(probability_excited)

        # ground probability formula
        probability_ground = np.matmul(ground_state.transpose(), psi.conjugate())
        probability_ground = abs(np.sum(probability_ground)) ** 2
        end_probability_ground.append(probability_ground)

        # norm of the psi matrix
        norm = abs(np.sum(psi * psi.conjugate()))
        if dimension >= 3:
            third_state = np.array(eigenpsi[:, [2]])

            # third state probability formula
            probability_third = np.matmul(third_state.transpose(), psi.conjugate())
            probability_third = abs(np.sum(probability_third)) ** 2
            end_probability_third.append(probability_third)
            # truncate
            if abs(probability_excited - 1) < amp / (10 * 3):
                print(f'time: {t(k)}, count: {k}')
                time_ms = t(k)
                leakage = 0
                for dim in range(2, dimension):
                    high_state = np.array(eigenpsi[:, [dim]])
                    # third state probability formula
                    probability_high = np.matmul(high_state.transpose(), psi.conjugate())
                    probability_high = abs(np.sum(probability_high)) ** 2
                    leakage += probability_high
                    # print('leakage:', leakage)
                print(f'leakage: {leakage}')
                print('\n')
                break

        else:
            third_state = 0
            probability_third = 0
            end_probability_third.append(probability_third)

# plot image
axis = [x * tau for x in range(len(end_probability_excited))]
fig, at = plt.subplots()
at.plot(axis, end_probability_excited, label='excited state')
at.plot(axis, end_probability_ground, label='ground state')
at.plot(axis, end_probability_third, label='third state')
at.set_xlabel('time, ns')
at.set_ylabel('probability')
at.set_title("Leakage graph")
at.legend(loc='lower left')
#plt.yscale("log")
plt.show()
