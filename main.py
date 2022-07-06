import math

import numpy as np
import matplotlib.pyplot as plt
import cmath

import pandas as pd


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
            self.matrix=None
        except AssertionError:
            print('PLease enter 0 for annihilation and 1 for creation')
            self.matrix=None

    def __repr__(self):
        return f'{self.matrix}'

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
amplitude_R = 0
time = 200
counts = 3000
tau = time / counts
alpha = 1

sigma = 3
mu = 0.3
beta = 30 / mu
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


def Rhabi_solution(k):
    for k in range(k + 1):
        # Start of psi calculation
        t = lambda k_: k_ * tau
        # Oscillatory amplitude (related to DRAG techinque)
        Omega = np.sqrt((omega_R - omega_01) ** 2 + amplitude_R ** 2)
        Rhabi = 1 - amplitude_R ** 2 / Omega ** 2 * np.sin(Omega * k * tau / 2) ** 2
        end_Rhabi.append(Rhabi)


end_probability_excited = []
end_probability_ground = []
end_probability_third = []
end_Rhabi = []
end_leakage = []
end_fidelity = []
end_times = []

amplitude_set = np.array([0.01])
amplitude_set = amplitude_set * 10 * 3.6
amplitude_set = amplitude_set.tolist()
# Finding eigenvalues and eigenvectors
Hamiltonian_0 = omega_01 * np.matmul(creation, annihilation) - mu / 2 * np.matmul(creation, annihilation) * (
        np.matmul(creation, annihilation) - I)
eigenenergy, eigenpsi = np.linalg.eig(Hamiltonian_0)

# Main cycle


for amp in amplitude_set:
    psi = init_psi(dimension)
    amplitude_R = amp
    time = 130 / amp
    tau = time / counts
    print(amplitude_set.index(amp), 'cycle, amp:', amp)
    for k in range(counts):
        # Start of psi calculation
        t = lambda k_: k_ * tau

        Omega_x_numerator = np.exp(-(t(k) - 0.5 + t_g) ** 2 / (2 * sigma ** 2)) - np.exp(
            -t_g ** 2 / (8 * sigma ** 2))
        Omega_x_denominatior = np.sqrt(2 * np.pi * sigma ** 2) * math.erf(t_g / (np.sqrt(8) * sigma)) - t_g * np.exp(
            -t_g ** 2 / (8 * sigma ** 2))
        Omega_x = alpha * amplitude_R * Omega_x_numerator / Omega_x_denominatior

        Omega_y_numerator = np.exp(-(t(k) - 0.5 + t_g) ** 2 / (2 * sigma ** 2)) * (t(k) - 0.5 * t_g)
        Omega_y_denominator = (sigma ** 2) * Omega_x_denominatior
        Omega_y = -beta * amplitude_R * (-Omega_y_numerator / Omega_y_denominator)

        oscillatory_part = Omega_x * np.cos(omega_R * (t(k + 1 / 2))) + Omega_y * np.sin(
            omega_R * (t(k + 1 / 2)))
        oscillatory_part_1st_derivative = -Omega_x * omega_R * np.sin(
            omega_R * (t(k + 1 / 2))) + Omega_y * omega_R * np.cos(omega_R * (t(k + 1 / 2)))
        oscillatory_part_2nd_derivative = -Omega_x * omega_R ** 2 * np.cos(
            omega_R * (t(k + 1 / 2))) - Omega_y * omega_R ** 2 * np.sin(omega_R * (t(k + 1 / 2)))

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
            if abs(probability_excited - 1) < amp / (10 * 3):
                print('time:', t(k), 'count:', k)
                end_times.append(t(k))
                leakage = 0
                for dim in range(2, dimension):
                    high_state = np.array(eigenpsi[:, [dim]])
                    # third state probability formula
                    probability_high = np.matmul(high_state.transpose(), psi.conjugate())
                    probability_high = abs(np.sum(probability_high)) ** 2
                    leakage += probability_high
                    # print('leakage:', leakage)
                end_leakage.append(leakage)
                break

        else:
            third_state = 0
            probability_third = 0
            end_probability_third.append(probability_third)

amplitude_R = 0.0085
Rhabi_solution(k)
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


# for amp in amplitude_set:
# amplitude_R=amp
# time = 3.1 / amp
# tau = time / counts
# fidelity=0
# for i in range(1, 7, 1):
#    psi_g = np.matmul(rotation_matrix, alpha_state(dimension, i))
#    probability = np.matmul(wave_function_calculation(alpha_state(dimension, i)).transpose(), psi_g.conjugate())
#    probability = abs(np.sum(probability)) ** 2
#    fidelity += 1 / 6 * probability
# end_fidelity.append(fidelity)
# print('amp:',amplitude_R)

# print('fidelity:', end_fidelity)
print('times:', end_times)
print('leakage:', end_leakage)

amps_harmonic = [0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.018, 0.014, 0.01]
times_harmonic = [23.482499999999998, 28.447666666666667, 32.251481481481484, 36.0375, 40.37380952380952,
                  46.20722222222222, 56.254666666666665, 70.525, 95.06666666666666, 144.87333333333333,
                  161.77407407407406, 210.65238095238095, 302.45666666666665]
leakages_harmonic = [0.047583990306042365, 0.0446278433526341, 0.04979107566586703, 0.042444563007531355,
                     0.027545869981192846, 0.014466045111689215, 0.014963140419702273, 0.007769732965873183,
                     0.004856940857545456, 0.0021219578486229864, 0.001742155749432928, 0.0009413733438241942,
                     0.00045073225985224353]

amps_DRAG = np.array([0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.018, 0.014, 0.01])
amps_DRAG = amps_DRAG * 10 * 3
amps_DRAG = amps_DRAG.tolist()
times_DRAG = [30.164814814814818, 36.934444444444445, 40.733333333333334, 45.75277777777777, 52.76349206349206,
              62.11111111111112, 75.31333333333333, 95.33333333333333, 128.94074074074075, 198.39444444444442,
              221.80246913580245, 291.468253968254]
leakages_DRAG = [0.03736012237436596, 0.033725729848171467, 0.02247082494879301, 0.014074110193456614,
                 0.013699714552306253, 0.009811150926293984, 0.0073055069997196185, 0.004694266829021064,
                 0.0024444111615832733, 0.0010983342728314884, 0.0008964416181296032, 0.00047733699931035156]
times_DRAG = [24.545524691358022, 30.16481481481481, 33.97119341563786, 38.26273148148148, 43.36772486772486,
              51.358024691358025, 62.1111111111111, 78.54166666666666, 106.28703703703704, 163.40277777777777,
              182.7623456790123, 238.76322751322746, 354.3703703703703]
leakages_DRAG = [0.03590847725341302, 0.03735999040648547, 0.03820440106584877, 0.02873768725404596,
                 0.01621071733574948, 0.014199073867334176, 0.009811264439498492, 0.00690441777028871,
                 0.0034817065187071625, 0.0016537223712833197, 0.0012968257526546733, 0.000764157973805926,
                 0.0003275457443872466]

DRAG04 = pd.read_csv(r'C:\Users\user\Desktop\sci_pics\DRAG, mu=0.4.csv')
timeDRAG = DRAG04['1']
timeDRAG = timeDRAG.tolist()
DRAG04 = DRAG04['0'].tolist()
DRAG03 = [
    0.000410683,
    0.0004877,
    0.000608415,
    0.000762359,
    0.000847101,
    0.000904151,
    0.001082597,
    0.001270423,
    0.001401482,
    0.001570653,
    0.001762067,
    0.001833766,
    0.002155761,
    0.002199288,
    0.002300221,
    0.002910981,
    0.003097042,
    0.003214701,
    0.003059539,
    0.003468283,
    0.003848702,
    0.004600581,
    0.004928758,
    0.004684306,
    0.004325207,
    0.004562894,
    0.005075713,
    0.005625154,
    0.006647806,
    0.006972311,
    0.007180335,
    0.007258016,
    0.007150232,
    0.006930945,
    0.007022579,
    0.007256654,
    0.00763524,
    0.00800752,
    0.008852712,
    0.009780681,
    0.010670467,
    0.011482894,
    0.012297807,
    0.013782637,
    0.013854394,
    0.013745615,
    0.014285269,
    0.014218121,
    0.013569816,
    0.0141092,
    0.013100248,
    0.01360709,
    0.012937639,
    0.012948432,
    0.013578053,
    0.014321098,
    0.014434642,
    0.014630399,
    0.015320404,
    0.016223769,
    0.017324291,
    0.01887281,
    0.020920362,
    0.021704846,
    0.022336682,
    0.023406837,
    0.024636062,
    0.027673655,
    0.027937386,
    0.028688169,
    0.029910374,
    0.032265207,
    0.032587645,
    0.033224108,
    0.035783015,
    0.035671855,
    0.035964728,
    0.038407165,
    0.037651247,
    0.038139939,
    0.039602987,
    0.038645484,
    0.041592159,
]
DRAG05 = pd.read_csv(r'C:\Users\user\Desktop\sci_pics\DRAG, mu=0.5.csv')
DRAG05 = DRAG05['0'].tolist()
#DRAG03[5:]=[x*0.6 for x in DRAG03[5:]]
# plot image
axis = timeDRAG
fig, at = plt.subplots()
at.plot(axis, DRAG03, label='mu=0.5')
at.plot(axis, DRAG04, label='mu=0.4')
at.plot(axis, DRAG05, label='mu=0.3')
# at.plot(axis, end_Rhabi, label='Rhabi solution')
at.set_xlabel('time, ms')
at.set_ylabel('probability')
at.set_title("Leakage graph")
at.legend(loc='lower left')
plt.yscale("log")
plt.show()
