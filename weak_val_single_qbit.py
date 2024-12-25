'''Weak values single qubit'''

import numpy as np

hbar = 2

# Creating 2x2 matrices for spin operators in x,y, and z
Sx = (hbar / 2) * np.array([0, 1, 1, 0]).reshape((2, 2))
Sy = (hbar / 2) * np.array([0, -1j, 1j, 0]).reshape((2, 2))
Sz = (hbar / 2) * np.array([1, 0, 0, -1]).reshape((2, 2))


def SpinOps(q: int):

    I = np.eye(2)
    for _ in range(q - 2):
        I = np.kron(I, np.eye(2))

    # Tensor product
    SxTI = np.kron(Sx, I)
    SyTI = np.kron(Sy, I)
    SzTI = np.kron(Sz, I)

    return SxTI, SyTI, SzTI


def initialize_state(num_qubits):
    # Define the single qubit |0> state
    zero_state = np.array([1, 0], dtype=complex)

    # Initialize the state as the tensor product of |0> states
    state = zero_state
    for _ in range(1, num_qubits):
        state = np.kron(state, zero_state)

    return state


def entangled_state(num_qubits):
    # Define Pauli matrices and identity
    I = np.array([[1, 0], [0, 1]], dtype=complex)

    # Rotation matrices
    def rotation_matrix(theta, phi, lam):
        return np.array([
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
        ], dtype=complex)

    # CNOT gate (generalized for control=qubit i, target=qubit i+1)
    def cnot_matrix(num_qubits, control, target):
        # Initialize to identity
        CNOT = np.eye(2 ** num_qubits, dtype=complex)
        # Create indices for the control and target bits
        for i in range(2 ** num_qubits):
            if (i >> control) & 1 and not (i >> target) & 1:
                # Swap |control=1, target=0> <=> |control=1, target=1>
                swap_idx = i ^ (1 << target)
                CNOT[i, i], CNOT[i, swap_idx] = 0, 1
                CNOT[swap_idx, i], CNOT[swap_idx, swap_idx] = 1, 0
        return CNOT

    # Initialize state |0>^n
    state = initialize_state(num_qubits)

    # Apply random rotation to each qubit
    for i in range(num_qubits):
        theta, phi, lam = np.random.uniform(0, 2 * np.pi, 3)
        R = rotation_matrix(theta, phi, lam)
        # Expand R to act on the i-th qubit
        U = 1
        for j in range(num_qubits):
            if j == i:
                U = np.kron(U, R)
            else:
                U = np.kron(U, I)
        state = U @ state

    # Apply CNOT gates between pairs of qubits
    for i in range(num_qubits - 1):
        CNOT = cnot_matrix(num_qubits, i, i + 1)
        state = CNOT @ state

    #TODO - return how entangled the state is (concurrence)
    # may need to add a knob to control the entanglement level
    return state


def sepstate_custom(theta=None, phi=None):
    """
       Generates a random pure quantum state on the Bloch sphere.

       Returns:
           np.ndarray: A 2D complex vector representing the quantum state |ψ⟩.
       """
    # Generate random angles for the Bloch sphere
    if theta is None:
        theta = np.arccos(1 - 2 * np.random.rand()) # Uniformly distributed theta in [0, π]

    if phi is None:
        phi = 2 * np.pi * np.random.rand()  # Uniformly distributed phi in [0, 2π)

    # Construct the state |ψ⟩
    state = np.array([
        np.cos(theta / 2),  # Amplitude of |0⟩
        np.exp(1j * phi) * np.sin(theta / 2)  # Amplitude of |1⟩
    ])
    return state


def SpinOpsR(q: int):
    # The same as SpinOps() but we reverse the order of the tensor product since this is for the 2nd qubit (regardless
    # of total number of qubits)

    I = np.eye(2)

    RxTI = np.kron(I, Sx)
    RyTI = np.kron(I, Sy)
    RzTI = np.kron(I, Sz)

    for _ in range(q - 2):
        RxTI = np.kron(RxTI, I)
        RyTI = np.kron(RyTI, I)
        RzTI = np.kron(RzTI, I)

    return RxTI, RyTI, RzTI

def WeakValue(j, g, S):
    """ Computes weak values for j and g evolution states (at the same step).
  :param j - forward evolved i state
  :param g - backward evolved f state
  :param S - spin operator (for left or right)
  :return Wreal - list of real part of computed weak values
  :return Wimag - list of imaginary part of computed weak values
  :return g_dot_j - dot product of forward evolved j and backward evolved f
                    to be checked for every step
  """

    Wreal = []  # Array for weak value before the gate (real part)
    Wimag = []  # Array for weak value before the gate (imaginary part)
    W_complex = []

    g_dot_j = np.around(np.inner(np.conj(g), j), decimals=7)

    for w in range(3):  # To generate three components for weak values for each coordinate axis, x, y, and, z
        # Using np.conjugate and np.dot, we can code the formula for weak values. S[w] allows us to use each spin operator in
        # order through indices. S[0] is the x-component, for instance.
        try:

            W = np.inner(np.conj(g), S[w].dot(j)) / g_dot_j  # Weak value

        except ZeroDivisionError as e:
            print("Error: Cannot divide by zero")
            W = float('inf')
        Wreal.append(W.real)  # Appending the real part of W
        Wimag.append(W.imag)  # Appending the imaginary part of W
        W_complex.append(W)

    return Wreal, Wimag, g_dot_j, W_complex

# ___________|i> = (|0> + |1>) / √2
# ____________________↓
# __________________/__\
# ________________ωa____ωb
# ______________/__________\
# ____________|ψa>___tensor_____|ψb>


##### |i> is a 2 qubit entangled state ######
# i = entangled_state(num_qubits=2) # entangled 2 qubit state
i = np.array([0, 0.6, -.8, 0])
# print(f"i: {i}")

# list of theta and phi combinations for up,down for x,y and z directions on the Bloch sphere separately
list_angles_z = [(0, 0), (np.pi, 0)]
list_angles_y = [(np.pi / 2, 0), (3 * np.pi / 2, 0)]
list_angles_x = [(np.pi / 2, 0), (np.pi / 2, np.pi)]
list_angles = [list_angles_x, list_angles_y, list_angles_z]

for angles in list_angles:
    theta_a, phi_a = angles[0]
    theta_b, phi_b = angles[1]
    psi_a = sepstate_custom(theta=theta_a, phi=phi_a)
    psi_b = sepstate_custom(theta=theta_b, phi=phi_b)
    f = np.kron(psi_a, psi_b)
    Wreal_a, Wimag_a, _, W_complex_a = WeakValue(i, f, SpinOps(q=2))
    Wreal_b, Wimag_b, _, W_complex_b = WeakValue(i, f, SpinOpsR(q=2))

    print("Sx | Sy | Sz")
    print(f"W_complex_a: {W_complex_a}")
    print(f"W_complex_b: {W_complex_b}")

    for i in range(3):
        print(f"Real W_complex_a[{i}]/W_complex_b[{i}]: {W_complex_a[i].real / W_complex_b[i].real}")

    for i in range(3):
        print(f"Imaginary W_complex_a[{i}]/W_complex_b[{i}]: {W_complex_a[i].imag / W_complex_b[i].imag}")

    #Calcualte magnitudes or real parts of the weak values
    norm_a = np.linalg.norm([val.real for val in W_complex_a])
    norm_b = np.linalg.norm([val.real for val in W_complex_b])

    #Compute probability of outcome
    prob_outcome = np.abs(np.inner(np.conj(f), i))**2

    print(f"Norm of W_a Real: {norm_a}")
    print(f"Norm of W_b Real: {norm_b}")
    print(f"Ratio of real norms: {norm_a / norm_b}")

    print(f"Probability of outcome: {prob_outcome}\n")