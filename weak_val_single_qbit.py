'''Weak values single qubit'''

import numpy as np
import matplotlib.pyplot as plt
from weak_values_legacy import *
from IPython.display import display, Math

hbar = 2

# Creating 2x2 matrices for spin operators in x,y, and z
Sx = (hbar / 2) * np.array([0, 1, 1, 0]).reshape((2, 2))
Sy = (hbar / 2) * np.array([0, -1j, 1j, 0]).reshape((2, 2))
Sz = (hbar / 2) * np.array([1, 0, 0, -1]).reshape((2, 2))


def angle_between_weak_values(W_a, W_b):
    """
    Computes the angle between the W_a and W_b three vectors.

    Parameters:
    W_a (list of complex): Weak values vector a.
    W_b (list of complex): Weak values vector b.

    Returns:
    float: The angle between W_a and W_b in radians.
    """
    # Convert to numpy arrays
    # W_a = np.array(W_a)
    # W_b = np.array(W_b)

    # Compute the dot product
    dot_product = np.vdot(W_b, W_a)

    # Compute the magnitudes
    magnitude_W_a = np.linalg.norm(W_a)
    magnitude_W_b = np.linalg.norm(W_b)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_W_a * magnitude_W_b)

    # Compute the angle in radians
    radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degrees = radians * (180 / np.pi)

    return np.round(degrees,6)


# Define standard basis states in terms of Bloch sphere angles (θ, φ)
BASIS_DICT = {
    "Z+": (0, 0),       # |0⟩
    "Z-": (np.pi, 0),   # |1⟩
    "X+": (np.pi/2, 0), # |+⟩
    "X-": (np.pi/2, np.pi), # |−⟩
    "Y+": (np.pi/2, np.pi/2), # |i⟩
    "Y-": (np.pi/2, -np.pi/2) # |-i⟩
}


def normalize_angle(angle):
    """Normalize an angle to the range [0, 2π)."""
    return angle % (2 * np.pi)


def generate_bloch_sweep(start_basis, end_basis, step=np.pi/90, halfway=False):
    """
    Generates θ and φ values for sweeping from `start_basis` to `end_basis`,
    and optionally continuing until reaching the opposite of the start basis.

    Parameters:
    - start_basis: str, the name of the starting basis ("Z+", "X-", etc.)
    - end_basis: str, the name of the target basis ("X+", "Y-", etc.)
    - step: Increment for sweeping (default π/90)
    - halfway: bool, whether to stop at `end_basis` (True) or continue to full sweep (False)

    Returns:
    - List of (θ, φ) pairs along the trajectory
    """
    if start_basis not in BASIS_DICT or end_basis not in BASIS_DICT:
        raise ValueError(f"Invalid basis names. Choose from {list(BASIS_DICT.keys())}.")

    theta_start, phi_start = BASIS_DICT[start_basis]
    theta_end, phi_end = BASIS_DICT[end_basis]

    # Calculate halfway values as the midpoint between start and end basis
    theta_halfway = (theta_start + theta_end) / 2
    phi_halfway = (phi_start + phi_end) / 2

    # If halfway is True, stop at `end_basis`, otherwise continue to full range
    if halfway:
        theta_end, phi_end = theta_halfway, phi_halfway

    angles = []
    theta_change = False
    phi_change = False

    # Normalize angles to [0, 2π)
    phi_start = normalize_angle(phi_start)
    phi_end = normalize_angle(phi_end)

    # Determine the number of steps needed
    theta_range = abs(theta_end - theta_start)
    phi_range = min(abs(phi_end - phi_start), 2 * np.pi - abs(phi_end - phi_start))

    if theta_range > 0 and phi_range > 0:
        num_steps = max(1, int(max(theta_range, phi_range) / step))
        theta_change = True
        phi_change = True
    elif theta_range > 0:
        num_steps = max(1, int(theta_range / step))
        theta_change = True
    else:
        num_steps = max(1, int(phi_range / step))
        phi_change = True

    # Generate theta and phi values based on whether they change or remain constant
    if theta_range > 0:
        theta_values = np.linspace(theta_start, theta_end, num_steps)
    else:
        theta_values = np.full(num_steps, theta_start)

    if phi_range > 0:
        phi_values = np.linspace(phi_start, phi_end, num_steps)
    else:
        phi_values = np.full(num_steps, phi_start)

    for t, p in zip(theta_values, phi_values):
        angles.append((t, p))

    # Ensure the last state exactly matches the expected final values
    angles[-1] = (theta_end, phi_end)

    return angles, [round(theta_start, 4), round(theta_end, 4), round(step, 4)], [round(phi_start, 4), round(phi_end, 4), round(step, 4)], theta_change, phi_change


# ___________|i> = (|0> + |1>) / √2
# ____________________↓
# __________________/__\
# ________________ωa____ωb
# ______________/__________\
# ____________|ψa>___tensor_____|ψb>


##### |i> is a 2 qubit entangled state ######
# i = entangled_state(num_qubits=2) # entangled 2 qubit state
i = np.array([0+0j, 0.6+0j, -.8+0j, 0+0j])

dict_vals_sweep = {"Real": {0: [], 1: [], 2: []}, "Imaginary": {0: [], 1: [], 2: []}, "Prob": [], "Angle": []}

# Example usage: Sweep from |0⟩ (Z-basis) to |+⟩ (X-basis) and back
angles_plus, theta_list, phi_list, theta_change, phi_change = generate_bloch_sweep("X+", "Y-", step=np.pi/(1 * 90), halfway=False)
angles_minus, theta_list_minus, phi_list_minus, _, _ = generate_bloch_sweep("X+", "Y-", step=np.pi/(1 * 90), halfway=False)

print(f"theta_list: {theta_list}")
print(f"theta_list_minus: {theta_list_minus}")
print(f"phi_list: {phi_list}")
print(f"phi_list_minus: {phi_list_minus}")

# print(f"angles_plus: {len(angles_plus)}")
# print(f"angles_minus: {len(angles_minus)}")

Wa_real = []
Wa_imag = []
Wb_real = []
Wb_imag = []
# Print first few angles
for c, (theta, phi) in enumerate(angles_plus):
    print(f"θ: {theta * 180 / np.pi:.3f}, φ: {phi * 180 / np.pi:.3f}")

    theta_minus, phi_minus = angles_minus[c]
    print(f"θ_minus: {theta_minus * 180 / np.pi:.3f}, φ_minus: {phi_minus * 180 / np.pi:.3f}")
    psi_a = sepstate_custom(theta=theta, phi=phi)
    # print(f"psi_a: {psi_a}")
    psi_b = sepstate_custom(theta=np.pi/2, phi=0) # theta=theta_minus, phi=phi_minus
    # print(f"psi_b: {psi_b}")
    f = np.kron(psi_a, psi_b)
    # print(f"f: {f}")
    Wreal_a, Wimag_a, _, W_a = WeakValue(i, f, SpinOps(q=2))
    Wreal_b, Wimag_b, _, W_b = WeakValue(i, f, SpinOpsR(q=2))

    Wa_real.append(Wreal_a)
    Wa_imag.append(Wimag_a)
    Wb_real.append(Wreal_b)
    Wb_imag.append(Wimag_b)

    # print(f"Basis: {basis}")
    # print(f"psi_a: {[np.round(comp, 3) for comp in psi_a]}")
    # print(f"psi_b: {[np.round(comp, 3) for comp in psi_b]}")

    print("Sx | Sy | Sz")
    # print(f"W_a: {[np.round(comp, 3) for comp in W_a]}")
    # print(f"W_b: {[np.round(comp, 3) for comp in W_b]}")
    print(f"W_a: {W_a}")
    print(f"W_b: {W_b}")

    for c in range(3):
        real = W_a[c].real / W_b[c].real
        imag = W_a[c].imag / W_b[c].imag
        # real = W_b[c].real / W_a[c].real
        # imag = W_b[c].imag / W_a[c].imag
        dict_vals_sweep["Real"][c].append(real)
        dict_vals_sweep["Imaginary"][c].append(imag)
        print(f"\033[32m Real W_a[{c}]/W_b[{c}]: {real} \033[0m")
        print(f"\033[32m Imaginary W_a[{c}]/W_b[{c}]: {imag} \033[0m")

    #Calcualte magnitudes or real parts of the weak values
    norm_a = np.linalg.norm([val.real for val in W_a])
    norm_b = np.linalg.norm([val.real for val in W_b])

    #Compute probability of outcome
    prob_outcome = np.round(np.abs(np.inner(np.conj(f), i)) ** 2, 3)
    dict_vals_sweep["Prob"].append(prob_outcome)

    print(f"Norm of W_a Real: {norm_a}")
    print(f"Norm of W_b Real: {norm_b}")
    # print(f"Complex Dot product of W_a and W_b: {np.vdot(W_b, W_a)}")
    print(f"Ratio of real norms: {norm_a / norm_b}")

    print(f"Probability of outcome: {np.round(prob_outcome, 3)}\n")
    print(f"Complex angle between W_a and W_b: {angle_between_weak_values(W_a, W_b)}\n")


# Plot values from dict_vals_sweep against theta
ffig, ax = plt.subplots(3, 2, figsize=(15, 15))
colors = ["red", "blue", "green"]
labels = ["X", "Y", "Z"]

for i in range(3):
    # First column: plot dict_vals_sweep["Real"][i] and dict_vals_sweep["Imaginary"][i]
    if theta_change:
        ax[i, 0].plot(np.linspace(theta_list[0] * 180/np.pi, theta_list[1] * 180/np.pi, len(dict_vals_sweep["Real"][i])), dict_vals_sweep["Real"][i], color=colors[i], label=f"Real {labels[i]}")
        ax[i, 0].plot(np.linspace(theta_list[0] * 180/np.pi, theta_list[1] * 180/np.pi, len(dict_vals_sweep["Imaginary"][i])), dict_vals_sweep["Imaginary"][i], color=colors[i], linestyle="--", label=f"Imaginary {labels[i]}")
        ax[i, 0].set_xlabel("Theta")
    elif phi_change:
        ax[i, 0].plot(np.linspace(phi_list[0] * 180/np.pi, phi_list[1] * 180/np.pi, len(dict_vals_sweep["Real"][i])), dict_vals_sweep["Real"][i], color=colors[i], label=f"Real {labels[i]}")
        ax[i, 0].plot(np.linspace(phi_list[0] * 180/np.pi, phi_list[1] * 180/np.pi, len(dict_vals_sweep["Imaginary"][i])), dict_vals_sweep["Imaginary"][i], color=colors[i], linestyle="--", label=f"Imaginary {labels[i]}")
        ax[i, 0].set_xlabel("Phi")
    ax[i, 0].set_ylabel("$W_a/W_b$")
    ax[i, 0].legend()
    ax[i, 0].grid()

    # Second column: plot 1/dict_vals_sweep["Real"][i] and 1/dict_vals_sweep["Imaginary"][i]
    if theta_change:
        ax[i, 1].plot(np.linspace(theta_list[0] * 180/np.pi, theta_list[1] * 180/np.pi, len(dict_vals_sweep["Real"][i])), 1/np.array(dict_vals_sweep["Real"][i]), color=colors[i], label=f"Real {labels[i]}")
        ax[i, 1].plot(np.linspace(theta_list[0] * 180/np.pi, theta_list[1] * 180/np.pi, len(dict_vals_sweep["Imaginary"][i])), 1/np.array(dict_vals_sweep["Imaginary"][i]), color=colors[i], linestyle="--", label=f"Imaginary {labels[i]}")
        ax[i, 1].set_xlabel("Theta")
    elif phi_change:
        ax[i, 1].plot(np.linspace(phi_list[0] * 180/np.pi, phi_list[1] * 180/np.pi, len(dict_vals_sweep["Real"][i])), 1/np.array(dict_vals_sweep["Real"][i]), color=colors[i], label=f"Real {labels[i]}")
        ax[i, 1].plot(np.linspace(phi_list[0] * 180/np.pi, phi_list[1] * 180/np.pi, len(dict_vals_sweep["Imaginary"][i])), 1/np.array(dict_vals_sweep["Imaginary"][i]), color=colors[i], linestyle="--", label=f"Imaginary {labels[i]}")
        ax[i, 1].set_xlabel("Phi")
    ax[i, 1].set_ylabel("$W_b/W_a$")
    ax[i, 1].legend()
    ax[i, 1].grid()

plt.tight_layout()
plt.show()

# Plot weak values
#_ = plot_weak_values(Wa_real, Wa_imag, Wb_real, Wb_imag, plot_quiver=True)
