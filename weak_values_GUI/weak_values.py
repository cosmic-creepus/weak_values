import gc
import csv
import time
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots

# importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import cmath as cm  # Libraries that will be needed for computation with complex numbers
from scipy.optimize import minimize, curve_fit
from sympy import *

hbar = 2


def Unitary(n_hat, theta=0, right=False):
    """
    Single unitary qubit gate.

    :param theta:
    :param n_hat: 3-vector that defines the qubit axis of rotation
    :param right: if True apply unitary gate to right qubit
    :return: U and U inverse
    """

    # Make n_hat a unit vector
    n_hat = np.array(n_hat) / np.linalg.norm(n_hat)

    # 2x2 Identity
    I = np.eye(2)

    # Creating 2x2 matrices for U and U inverse
    # Unitary mat of single qubit rotation around y and z axis on Bloch sphere
    # Umat = np.array([(np.cos(theta / 2)), -np.exp(1j * lam) * (np.sin(theta / 2)),
    #                 np.exp(1j * phi) * (np.sin(theta / 2)), np.exp(1j * (lam + phi)) * (np.cos(theta / 2))]).reshape(2, 2)

    # Unitary rotation about an arbitrary axis n
    nx, ny, nz = tuple(n_hat)
    sigma_n = np.array([nz, nx - 1j * ny, nx + 1j * ny, -nz]).reshape(2, 2)

    Umat = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * sigma_n
    Umat_inv = np.linalg.inv(Umat)
    # print(f"Umat {Umat}")

    # Tensor product of U X I and U^-1 X I since we're working with 2-qubit states
    if right:
        U = np.kron(I, Umat)
        Uinv = np.kron(I, Umat_inv)
    else:
        U = np.kron(Umat, I)
        Uinv = np.kron(Umat_inv, I)

    # U X I and U^-1 X I
    return U, Uinv


def SpinOps():
    # Creating 2x2 matrices for spin operators in x,y, and z
    Sx = (hbar / 2) * np.array([0, 1, 1, 0]).reshape((2, 2))
    Sy = (hbar / 2) * np.array([0, -1j, 1j, 0]).reshape((2, 2))
    Sz = (hbar / 2) * np.array([1, 0, 0, -1]).reshape((2, 2))

    I = np.eye(2)

    # Tensor product
    SxTI = np.kron(Sx, I)
    SyTI = np.kron(Sy, I)
    SzTI = np.kron(Sz, I)

    return SxTI, SyTI, SzTI


def SpinOpsR():
    # The same as SpinOps() but we reverse the order of the tensor product since this is for the right wing
    Sx = (hbar / 2) * np.array([0, 1, 1, 0]).reshape((2, 2))
    Sy = (hbar / 2) * np.array([0, -1j, 1j, 0]).reshape((2, 2))
    Sz = (hbar / 2) * np.array([1, 0, 0, -1]).reshape((2, 2))

    I = np.eye(2)

    RxTI = np.kron(I, Sx)
    RyTI = np.kron(I, Sy)
    RzTI = np.kron(I, Sz)

    return RxTI, RyTI, RzTI


# To generate a table
class ListTable(list):

    def __init__(self, table_header, table_data):
        super().__init__()
        self.table_header = table_header
        self.table_data = table_data

    def repr_html(self):
        html = ["<table>"]

        for data in [[self.table_header], self.table_data]:
            for row in data:
                html.append("<tr>")
                for col in row:
                    html.append("<td>{0}</td>".format(col))

                html.append("</tr>")
        html.append("</table>")
        html.append("</body>")
        return ''.join(html)


def nestrs(n):  # Our function for the nested rSWAP gate and its inverse
    """
    The unitary time evolution operator
    :param n: root of the swap
    :return: evolution operator and its inverse
    """
    nrs = np.array(
        [[1, 0, 0, 0], [0, (1 + np.exp((np.pi / 2 ** n) * 1j)) / 2, (1 - np.exp((np.pi / 2 ** n) * 1j)) / 2, 0],
         [0, (1 - np.exp((np.pi / 2 ** n) * 1j)) / 2, (1 + np.exp((np.pi / 2 ** n) * 1j)) / 2, 0], [0, 0, 0, 1]])
    # nrs = np.array([[1,0,0,0],[0,(1-np.exp((np.pi/2**n)*1j))/2,(1+np.exp((np.pi/2**n)*1j))/2,0],[0,(1+np.exp((np.pi/2**n)*1j))/2,(1-np.exp((np.pi/2**n)*1j))/2,0],[0,0,0,1]])

    nrsinv = np.array(
        [[1, 0, 0, 0], [0, (1 + np.exp((-np.pi / 2 ** n) * 1j)) / 2, (1 - np.exp((-np.pi / 2 ** n) * 1j)) / 2, 0],
         [0, (1 - np.exp((-np.pi / 2 ** n) * 1j)) / 2, (1 + np.exp((-np.pi / 2 ** n) * 1j)) / 2, 0], [0, 0, 0, 1]])
    return nrs, np.linalg.inv(nrs)


def entstate():
    initial = []  # Array for 2-qubit state
    msumi = 0  # msumi will be the magnitude squared of the unnormalized entangled state
    for l in range(4):  # To generate the four coefficients for 2-qubits
        amp = round(rd.random(), 2)  # Amplitude between 0 to 1
        com = round(rd.uniform(0, 2 * np.pi), 2)  # Argument between 0 to 2pi for complex phase
        comamp = amp * cm.exp(com * 1j)  # Combine amp and com to create a complex amplitude
        msumi += amp ** 2  # Summing square of coefficients of unnormalized state
        initial.append(comamp)  # Appending the complex amplitude as an element of the array 'initial'
        # Elements 0,1,2,3 correspond to ++, +-, -+, and --, respectively.

    initnorm = initial / np.sqrt(msumi)  # Normalizing our state
    return initnorm


# Mostly follows the same structure used for preparing the initial state
def sepstate():
    initial1 = []
    initial2 = []  # Both of these arrays will be single qubit state used to generate a separable 2-qubit state
    for l in range(2):  # To generate the two coefficients for each qubit
        amp1 = round(rd.random(), 2)
        amp2 = round(rd.random(), 2)
        com1 = round(rd.uniform(0, 2 * np.pi), 2)
        com2 = round(rd.uniform(0, 2 * np.pi), 2)
        comamp1 = amp1 * cm.exp(com1 * 1j)
        comamp2 = amp2 * cm.exp(com2 * 1j)
        initial1.append(comamp1)
        initial2.append(comamp2)  # Same procedure as the entstate function for this part

    sep = np.kron(initial1, initial2)  # Tensor product the two qubits to make separable final state

    msumf = 0  # msumf will be the magnitude squared of the unnormalized separable state
    for k in range(4):
        msumf += abs(sep[k]) ** 2  # Summing square of coefficients of unnormalized state

    sepnorm = sep / np.sqrt(msumf)  # Normalizing our state
    return sepnorm


# Using the concurrence formula to check if the state is entangled or not. Implemented conc<10^-16 rather than conc=0 to account
# for my python truncating at 8 digits after the decimal place. Larger value than 10^-16 needed if less digits after decimal.
def entchecker(state):
    conc = 2 * abs(state[0] * state[3] - state[1] * state[2])  # defining concurrence
    # print(conc)
    if conc < 10 ** -15:  # If concurrence is less than 10^-15 (accounting for precision error), we conclude state is separable
        return False  # Separable
    else:
        return True  # Entangled


def rotate_vector_to_parallel(vector, target_vector):
    """ Computes the rotation matrix to rotate a vector to become parallel to another vector
     :param vector - vector to be rotated to parallel to target_vector
     :param target_vector - target vector
     :return rotation_matrix - rotation matrix
    """
    # Ensure the input vectors are NumPy arrays
    vector = np.array(vector)
    target_vector = np.array(target_vector)

    # Normalize the input vectors
    vector = vector / np.linalg.norm(vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Calculate the angle between the two vectors
    angle = np.arccos(np.dot(vector, target_vector))

    # Calculate the rotation axis
    rotation_axis = np.cross(vector, target_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    # print(f"Rotation vec: {rotation_axis}")

    # Create a rotation matrix using the Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = (
            cos_angle * np.identity(3) +
            (1 - cos_angle) * np.outer(rotation_axis, rotation_axis) +
            sin_angle * np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]])
    )

    # Rotate the vector
    # rotated_vector = np.dot(rotation_matrix, vector)

    return rotation_matrix


def extract_xyz_coordinate(vec):
    """ Extract x, y, z coordinates from a list of vectors and returns
    3 lists that each contain all x, all y and all z coordinates """
    xs = []
    ys = []
    zs = []
    for v in vec:
        xs.append(v[0])
        ys.append(v[1])
        zs.append(v[2])

    return xs, ys, zs


def fit_ellipse_to_points(points):
    """
    Fits an ellipse to given points and extracts (center_x, center_y), semi_major_axis, semi_minor_axis and
    angle of rotation
    :param points: weak value vector point from rotated ellipse
    :return: (center_x, center_y), semi_major_axis, semi_minor_axis, angle
    """

    def objective_function(params, *args):
        a, b, c, d, f, g = params
        x, y = args

        func = a * x ** 2 + 2 * b * x * y + c * y ** 2 + 2 * d * x + 2 * f * y + g
        return rd.choice(func)

    # Extract x and y coordinates from the input points
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])

    # Initial guess for ellipse parameters
    initial_params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Use the least squares method to fit an ellipse
    optimized_params = minimize(
        objective_function, initial_params, args=(x, y),
        method='L-BFGS-B', bounds=[(0, None) for _ in range(6)]
    )

    # Extract optimized ellipse parameters
    a, b, c, d, f, g = optimized_params.x

    # Calculate the center and axes of the fitted ellipse
    center_x = (c * d - b * f) / (b ** 2 - a * c)
    center_y = (a * f - b * d) / (b ** 2 - a * c)
    semi_minor_axis = np.sqrt(2 / (a + c + np.sqrt((a - c) ** 2 + b ** 2)))
    semi_major_axis = np.sqrt(2 / (a + c - np.sqrt((a - c) ** 2 + b ** 2)))

    # Calculate the angle of the major axis
    angle_radians = 0.5 * np.arctan(2 * b / (a - c))

    # Return the center, semi-major axis, semi-minor axis, and angle
    return (center_x, center_y), semi_major_axis, semi_minor_axis, np.degrees(angle_radians)


def check_points_on_ellipse(points, center, semi_major_axis, semi_minor_axis, foci, tolerance=0.01):
    """
    Check if a set of points lies on or near an ellipse.

    Parameters:
    - points: List of lists, where each sublist contains [x, y] coordinates of a point.
    - center: Tuple or list representing the center of the ellipse (h, k).
    - semi_major_axis: Length of the semi-major axis (a).
    - semi_minor_axis: Length of the semi-minor axis (b).
    - tolerance: Tolerance for considering points on the ellipse. Default is 0.1.

    Returns:
    - List of booleans indicating whether each point is on or near the ellipse.
    """

    h, k, _ = set(center)

    def ellipse_equation(x, y, semi_major_axis, semi_minor_axis, center):
        h, k, _ = set(center)
        return ((x - h) ** 2) / (semi_major_axis ** 2) + ((y - k) ** 2) / (semi_minor_axis ** 2)

    ellipse_point_status = []
    ellipse_one = []
    on_ellipse = []
    not_on_ellipse = 0
    for c, point in enumerate(points):
        ellipse_one.append(
            ellipse_equation(x=point[0], y=point[1], semi_major_axis=semi_major_axis, semi_minor_axis=semi_minor_axis,
                             center=center))
        ellipse_point_status.append(np.isclose(ellipse_one[-1], 1, rtol=tolerance))
        if not ellipse_point_status[-1]:
            not_on_ellipse += 1
        else:
            # record point on ellipse
            on_ellipse.append(c)

    is_an_ellipse = all(ellipse_point_status)
    if is_an_ellipse:
        print("All points belong to an ellipse")
    else:
        print(
            f"Not all points belong to an ellipse. {not_on_ellipse} out of {len(ellipse_point_status)} are not on ellipse")

    # Check if the locus of points for which the sum of the distances to two fixed points is constant.
    # sum_dist = []
    # for point in points:
    #     dist_f1 = np.sqrt((point[0] - foci[0][0]) ** 2 + (point[1] - foci[0][1]) ** 2)
    #     dist_f2 = np.sqrt((point[0] - foci[1][0]) ** 2 + (point[1] - foci[1][1]) ** 2)
    #     dist = dist_f1 + dist_f2
    #     sum_dist.append(dist)
    #
    # dist_check1 = list(.95 * dist <= np.array(sum_dist))
    # dist_check2 = list(1.05 * dist >= np.array(sum_dist))
    #
    # if dist_check1 and dist_check2:
    #     print(f"Ellipse PASSED sum test - it is an ellipse")
    # else:
    #     print(f"Ellipse did not passed sum test - it is NOT an ellipse")

    return is_an_ellipse


def calc_normal_vector(p1, p2, p3):
    """
    Calculate normal to a plane formed by 3 points.

    :param p1: common point
    :param p2: other 2nd point
    :param p3: other 3rd point
    :return: normal as a numpy array
    """

    vec1 = [p1[c] - p2[c] for c in range(len(p1))]
    vec2 = [p1[c] - p3[c] for c in range(len(p1))]

    cross_vector = np.cross(vec1, vec2)
    magnitude = np.linalg.norm(cross_vector)

    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector.")

    normalized_vector = cross_vector / magnitude

    return normalized_vector


def plot_weak_values(weak_values_all_left: list = None,
                     weak_values_all_right: list = None,
                     weak_values_all_left_imag: list = None,
                     weak_values_all_right_imag: list = None,
                     weak_values: list = None,
                     plot_quiver: bool = False,
                     plot_plane: bool = False,
                     plot_ellipse_axis: bool = False,
                     center_lines: bool = False,
                     plot_foci: bool = False,
                     show_plot: bool = False):
    if weak_values is None:
        weak_values = ["real", "imaginary"]
    ellipse_dict = {}

    real = [weak_values_all_left, weak_values_all_right]
    imag = [weak_values_all_left_imag, weak_values_all_right_imag]

    ellipse_data_r = find_dist_around_ellipse(weak_values_all_left, weak_values_all_right)
    ellipse_data_i = find_dist_around_ellipse(weak_values_all_left_imag, weak_values_all_right_imag)

    center_r = ellipse_data_r['center']
    center_i = ellipse_data_i['center']

    if len(weak_values) == 2:
        plot_vecs = [real, imag]
    else:
        if weak_values[0] == 'real':
            plot_vecs = [real]
        else:
            plot_vecs = [imag]

    order = 0
    final_rotated_values = {}
    traces = []
    for c, vec in enumerate(plot_vecs):

        print(f"\nProcessing data for {weak_values[order]}")
        ellipse_dict.update({weak_values[order]: ''})

        # creating dataset for plotting from weak value vectors
        xs_left_first, ys_left_first, zs_left_first = extract_xyz_coordinate([vec[0][0]])
        xs_right_first, ys_right_first, zs_right_first = extract_xyz_coordinate([vec[1][0]])

        # First point in the nth root swap (chosen by default as the left side)
        start_point_left = [xs_left_first, ys_left_first, zs_left_first]
        start_point_right = [xs_right_first, ys_right_first, zs_right_first]

        ellipse_dict[weak_values[order]] = {"start_point_left": start_point_left,
                                            "start_point_right": start_point_right}

        xs_left, ys_left, zs_left = extract_xyz_coordinate(vec[0])
        xs_right, ys_right, zs_right = extract_xyz_coordinate(vec[1])

        # creating the scatter plot
        traces.append(
            go.Scatter3d(x=xs_left, y=ys_left, z=zs_left, name='Left Real' if vec is real else 'Left Imaginary'))
        traces.append(
            go.Scatter3d(x=xs_right, y=ys_right, z=zs_right, name='Right Real' if vec is real else 'Right Imaginary'))

        # Avoid plot legend double labeling
        if c == 0:
            traces.append(go.Scatter3d(x=xs_left_first, y=ys_left_first, z=zs_left_first,
                                       name='First Left'))  # plot first step point for left
            traces.append(go.Scatter3d(x=xs_right_first, y=ys_right_first, z=zs_right_first,
                                       name='First Right'))  # plot first step point for right
        else:
            traces.append(go.Scatter3d(x=xs_left_first, y=ys_left_first, z=zs_left_first,
                                       name='First Left'))  # plot first step point for left
            traces.append(go.Scatter3d(x=xs_right_first, y=ys_right_first, z=zs_right_first,
                                       name='First Right'))  # plot first step point for right

        # Plane equation aX + bY + cZ = d Find plane equation coefficients by picking 3 points, and finding the
        # normal to the plane by crossing two vectors made from the 3 points
        p1 = vec[0][0]
        p2 = vec[0][-1]
        p3 = vec[-1][1]
        normal_vec = calc_normal_vector(p1=p1, p2=p2, p3=p3)
        ellipse_dict[weak_values[order]]['normal_vec'] = list(normal_vec)

        # create plane plot made by vector (tip points)
        a = normal_vec[0]
        b = normal_vec[1]
        c = normal_vec[2]
        d = a * p1[0] + b * p1[1] + c * p1[2]
        # mag_normal = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        # normalized_normal_vec = [a / mag_normal, b / mag_normal, c / mag_normal]

        xs_left_rot = []
        ys_left_rot = []
        zs_left_rot = []

        xs_right_rot = []
        ys_right_rot = []
        zs_right_rot = []

        # get rotation matrix for rotating ellipse(of weak value vectors) parallel to x-y plane
        rot_matrix_xy = rotate_vector_to_parallel(normal_vec, [0, 0, 1])

        # repackage rotated values, each in a new list (rotated to the same z coordinate (or parallel to x-y plane))
        weak_values_all_left_rotated = []
        weak_values_all_right_rotated = []
        for l in range(len(xs_left)):
            rotated_vec_left = np.matmul(rot_matrix_xy, np.array([xs_left[l], ys_left[l], zs_left[l]]))
            xs_left_rot.append(rotated_vec_left[0])
            ys_left_rot.append(rotated_vec_left[1])
            zs_left_rot.append(rotated_vec_left[2])
            weak_values_all_left_rotated.append(rotated_vec_left)

            rotated_vec_right = np.matmul(rot_matrix_xy, np.array([xs_right[l], ys_right[l], zs_right[l]]))
            xs_right_rot.append(rotated_vec_right[0])
            ys_right_rot.append(rotated_vec_right[1])
            zs_right_rot.append(rotated_vec_right[2])
            weak_values_all_right_rotated.append(rotated_vec_right)

        # get major and minor axis for rotated ellipse (parallel to x-y plane)
        xy_data = find_dist_around_ellipse(weak_values_all_left_rotated, weak_values_all_right_rotated)
        # ax.scatter(center[0], center[1], center[2], color='purple')  # plot center of current ellipse
        # ax.scatter(max_right[0], max_right[1], max_right[2], color='yellow')  # plot max right of current ellipse

        xs_left_rot_x = []
        ys_left_rot_x = []
        zs_left_rot_x = []

        xs_right_rot_x = []
        ys_right_rot_x = []
        zs_right_rot_x = []

        # get rotation matrix to rotate semi-major axis parallel to x-axis
        # print(f"max_right, center: {max_right, center}")
        vec_xy = [xy_data['max_right'][c] - xy_data['center'][c] for c in
                  range(3)]  # vector in x-y plane along the semi-major axis
        rot_matrix_x = rotate_vector_to_parallel(vec_xy, [1, 0, 0])

        # repackage rotated values above, each in a new list (rotated to the same z coordinate (or parallel to x-y plane))
        weak_values_all_left_rot_x = []
        weak_values_all_right_rot_x = []
        for l in range(len(xs_left_rot)):
            rotated_vec_left_x = np.matmul(rot_matrix_x, np.array([xs_left_rot[l], ys_left_rot[l], zs_left_rot[l]]))
            xs_left_rot_x.append(rotated_vec_left_x[0])
            ys_left_rot_x.append(rotated_vec_left_x[1])
            zs_left_rot_x.append(rotated_vec_left_x[2])
            weak_values_all_left_rot_x.append(rotated_vec_left_x)

            rotated_vec_right_x = np.matmul(rot_matrix_x, np.array([xs_right_rot[l], ys_right_rot[l], zs_right_rot[l]]))
            xs_right_rot_x.append(rotated_vec_right_x[0])
            ys_right_rot_x.append(rotated_vec_right_x[1])
            zs_right_rot_x.append(rotated_vec_right_x[2])
            weak_values_all_right_rot_x.append(rotated_vec_right_x)

        # weak_values_all_left_rot_x = np.around(weak_values_all_left_rot_x, decimals=3)
        # weak_values_all_right_rot_x = np.around(weak_values_all_right_rot_x, decimals=3)

        x_rot_data = find_dist_around_ellipse(weak_values_all_left_rot_x, weak_values_all_right_rot_x)

        # ax.scatter(center[0], center[1], center[2], color='purple')  # plot center of current ellipse
        # ax.scatter(max_right[0], max_right[1], max_right[2], color='black')  # plot max right of current ellipse
        if plot_foci:
            traces.append(
                go.Scatter3d(x=[x_rot_data['foci'][0][0]], y=[x_rot_data['foci'][0][1]], z=[x_rot_data['foci'][0][2]],
                             ))  # plot left focus of current ellipse
            traces.append(
                go.Scatter3d(x=[x_rot_data['foci'][1][0]], y=[x_rot_data['foci'][1][1]], z=[x_rot_data['foci'][1][2]],
                             ))  # plot right focus of current ellipse

        is_ellipse = check_points_on_ellipse(points=weak_values_all_left_rot_x + weak_values_all_right_rot_x,
                                             center=x_rot_data['center'], semi_major_axis=x_rot_data['major_axis'],
                                             semi_minor_axis=x_rot_data['minor_axis'], foci=x_rot_data['foci'],
                                             tolerance=.05)

        if plot_ellipse_axis:
            # plot ellipse semi-major axis
            x_sm, y_sm, z_sm = extract_xyz_coordinate([x_rot_data['max_left'], x_rot_data['max_right']])
            traces.append(go.Scatter3d(x=x_sm, y=y_sm, z=z_sm, mode='lines'))

            # plot ellipse semi-minor axis
            x_sm, y_sm, z_sm = extract_xyz_coordinate([x_rot_data['min_left'], x_rot_data['min_right']])
            traces.append(go.Scatter3d(x=x_sm, y=y_sm, z=z_sm, mode='lines'))

        if center_lines:
            # plot center to center line
            x_ctc, y_ctc, z_ctc = extract_xyz_coordinate([center_r, center_i])
            traces.append(go.Scatter3d(x=x_ctc, y=y_ctc, z=z_ctc, mode='lines'))
            x_cr, y_cr, z_cr = extract_xyz_coordinate([center_r, [0, 0, 0]])
            traces.append(go.Scatter3d(x=2 * np.array(x_cr), y=2 * np.array(y_cr), z=2 * np.array(z_cr), mode='lines'))
            x_ci, y_ci, z_ci = extract_xyz_coordinate([center_i, [0, 0, 0]])
            traces.append(go.Scatter3d(x=2 * np.array(x_ci), y=2 * np.array(y_ci), z=2 * np.array(z_ci), mode='lines'))

            # plot lines from center of ellipse to tip of double vector
            x_ttc_r, y_ttc_r, z_ttc_r = extract_xyz_coordinate([center_r, 2 * np.array(center_i)])
            x_ttc_i, y_ttc_i, z_ttc_i = extract_xyz_coordinate([center_i, 2 * np.array(center_r)])

            traces.append(go.Scatter3d(x=x_ttc_r, y=y_ttc_r, z=z_ttc_r))
            traces.append(go.Scatter3d(x=x_ttc_i, y=y_ttc_i, z=z_ttc_i))

        # ax.scatter(xs_left_rot_x, ys_left_rot_x, zs_left_rot_x, color='green' if vec is real else 'yellow')
        # ax.scatter(xs_right_rot_x, ys_right_rot_x, zs_right_rot_x, color='blue' if vec is real else 'brown')
        # ax.scatter([0], [0], [0], color='red')  # plot origin of x-y-z
        # ax.scatter(xs_left_first, ys_left_first, zs_left_first,
        #            color='magenta')  # plot first step point for left
        # ax.scatter(xs_right_first, ys_right_first, zs_right_first,
        #            color='yellow')  # plot first step point for right

        # get range for x and y for plane plot
        # x_min = int(min(min(xs_left), min(xs_right), xs_left_first[0], xs_right_first[0]))
        # x_max = int(max(max(xs_left), max(xs_right), xs_left_first[0], xs_right_first[0]))
        # y_min = int(min(min(ys_left), min(ys_right), ys_left_first[0]))
        # y_max = int(max(max(ys_left), max(ys_right), ys_left_first[0]))

        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)

        X, Y = np.meshgrid(x, y)
        Z = (d - a * X - b * Y) / c

        if plot_plane:
            traces.append(go.Surface(x=X, y=Y, z=Z, opacity=.5))

        # plot vectors
        if plot_quiver:
            for v in weak_values_all_right:
                traces.append(
                    go.Scatter3d(x=np.array([0, v[0]]), y=np.array([0, v[1]]), z=np.array([0, v[2]]), mode='lines'))
            for v in weak_values_all_left:
                traces.append(
                    go.Scatter3d(x=np.array([0, v[0]]), y=np.array([0, v[1]]), z=np.array([0, v[2]]), mode='lines'))

            for v in weak_values_all_right_imag:
                traces.append(
                    go.Scatter3d(x=np.array([0, v[0]]), y=np.array([0, v[1]]), z=np.array([0, v[2]]), mode='lines'))
            for v in weak_values_all_left_imag:
                traces.append(
                    go.Scatter3d(x=np.array([0, v[0]]), y=np.array([0, v[1]]), z=np.array([0, v[2]]), mode='lines'))

        final_rotated_values.update({weak_values[order]: [weak_values_all_left_rot_x, weak_values_all_right_rot_x]})

        order += 1

    # setting title and labels
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], name='Origin'))  # plot origin

    if show_plot:
        fig_p = go.Figure()
        for trace in traces:
            fig_p.add_trace(trace)

        fig_p.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        ### Important - comment/uncomment below 2 lines as needed per their own description ###
        # fig_p.write_html(os.getcwd() + "/plot.html", auto_open=False)  # Modify the html file with local server
        fig_p.write_html(os.getcwd() + "/weak_values_GUI/plot.html",
                         auto_open=False)  # Modify the html file with web server
        # fig_p.show() #opens the file on the local server

    return weak_values_all_left_rot_x, weak_values_all_right_rot_x, final_rotated_values, ellipse_dict, is_ellipse


def plot_3d_radius(weak_values_left, weak_values_right, weak_values_left_imag, weak_values_right_imag, show_plot=True):
    """
    Compute the radius of the complex weak values
    :param weak_values_all_left:
    :param weak_values_all_right:
    :param weak_values_all_left_imag:
    :param weak_values_all_right_imag:
    """

    x_weak_left, y_weak_left, z_weak_left = extract_xyz_coordinate(weak_values_left)
    x_weak_left_imag, y_weak_left_imag, z_weak_left_imag = extract_xyz_coordinate(weak_values_left_imag)

    x_weak_right, y_weak_right, z_weak_right = extract_xyz_coordinate(weak_values_right)
    x_weak_right_imag, y_weak_right_imag, z_weak_right_imag = extract_xyz_coordinate(weak_values_right_imag)

    radius_left_x = []
    radius_left_y = []
    radius_left_z = []
    radius_right_x = []
    radius_right_y = []
    radius_right_z = []

    for c in range(len(z_weak_left)):
        radius_left_x.append(np.sqrt(x_weak_left[c] ** 2 + x_weak_left_imag[c] ** 2))
        radius_left_y.append(np.sqrt(y_weak_left[c] ** 2 + y_weak_left_imag[c] ** 2))
        radius_left_z.append(np.sqrt(z_weak_left[c] ** 2 + z_weak_left_imag[c] ** 2))

        radius_right_x.append(np.sqrt(x_weak_right[c] ** 2 + x_weak_right_imag[c] ** 2))
        radius_right_y.append(np.sqrt(y_weak_right[c] ** 2 + y_weak_right_imag[c] ** 2))
        radius_right_z.append(np.sqrt(z_weak_right[c] ** 2 + z_weak_right_imag[c] ** 2))

    if show_plot:
        # creating figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Radius plot')
        ax.axis('scaled')

        plt.scatter(radius_left_x, radius_left_y, radius_left_z, color='green', marker='o', label='Left')
        plt.scatter(radius_right_x, radius_right_y, radius_right_z, color='blue', marker='o', label='Right')
        ax.scatter([0], [0], [0], color='red', marker='o', label='Origin')  # plot origin

        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        plt.legend()
        # plt.grid(True)
        plt.show()


def plot_complex_plane(weak_values_left, weak_values_right, weak_values_left_imag, weak_values_right_imag,
                       show_plot=True):
    """
    Compute the radius of the complex weak values
    :param weak_values_all_left:
    :param weak_values_all_right:
    :param weak_values_all_left_imag:
    :param weak_values_all_right_imag:
    """

    x_weak_left, y_weak_left, z_weak_left = extract_xyz_coordinate(weak_values_left)
    x_weak_left_imag, y_weak_left_imag, z_weak_left_imag = extract_xyz_coordinate(weak_values_left_imag)

    x_weak_right, y_weak_right, z_weak_right = extract_xyz_coordinate(weak_values_right)
    x_weak_right_imag, y_weak_right_imag, z_weak_right_imag = extract_xyz_coordinate(weak_values_right_imag)

    if show_plot:
        plt.scatter(x_weak_left, x_weak_left_imag, color='red', label='X left')
        plt.scatter(x_weak_right, x_weak_right_imag, color='pink', label='X right')

        plt.scatter(y_weak_left, y_weak_left_imag, color='blue', label='Y left')
        plt.scatter(y_weak_right, y_weak_right_imag, color='cyan', label='Y right')

        plt.scatter(z_weak_left, z_weak_left_imag, color='green', label='Z left')
        plt.scatter(z_weak_right, z_weak_right_imag, color='#88c999', label='Z right')

        plt.xlabel('real-axis')
        plt.ylabel('imaginary-axis')
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_axis_coordinate(weak_values_left, weak_values_right, weak_values_left_imag, weak_values_right_imag,
                         show_plot=True):
    """
    Compute the radius of the complex weak values
    :param weak_values_all_left:
    :param weak_values_all_right:
    :param weak_values_all_left_imag:
    :param weak_values_all_right_imag:
    """

    x_weak_left, y_weak_left, z_weak_left = extract_xyz_coordinate(weak_values_left)
    x_weak_left_imag, y_weak_left_imag, z_weak_left_imag = extract_xyz_coordinate(weak_values_left_imag)

    x_weak_right, y_weak_right, z_weak_right = extract_xyz_coordinate(weak_values_right)
    x_weak_right_imag, y_weak_right_imag, z_weak_right_imag = extract_xyz_coordinate(weak_values_right_imag)

    # # fit sine
    def func(theta, amp, freq, phase_shift, offset):
        return amp * np.sin(freq * theta + phase_shift) + offset

    theta_data = np.linspace(0, 2 * np.pi, len(x_weak_left + x_weak_right))
    # Perform the sine wave fit
    fit_params, covariance = curve_fit(func, theta_data, x_weak_left + x_weak_right, p0=[0.0, 1.0, 0.0, 0.0])
    print('Fit params - Amplitude: %5.3f | Freq: %5.3f | Phase shift %5.3f | Offset: %5.3f' % tuple(fit_params))
    print(f"Phase shift {fit_params[3] * 180 / np.pi}")

    x_axis_data = np.linspace(0, len(x_weak_left + x_weak_right), num=len(x_weak_left + x_weak_right))
    if show_plot:
        plt.scatter(x_axis_data, x_weak_left + x_weak_right, color='red', label='X left + right')
        # plt.scatter(np.linspace(0, len(x_weak_left), num=len(x_weak_right)), x_weak_right color='pink', label='X right')

        plt.scatter(x_axis_data, func(theta_data, *fit_params), marker='o', s=10, color='green',
                    label='fit: %5.3f * sin(%5.3f * theta + %5.3f) + %5.3f' % tuple(
                        fit_params))  # plt.scatter(y_weak_left, y_weak_left_imag, color='blue', label='Y left')
        # plt.scatter(y_weak_right, y_weak_right_imag, color='cyan', label='Y right')
        #
        # plt.scatter(z_weak_left, z_weak_left_imag, color='green', label='Z left')
        # plt.scatter(z_weak_right, z_weak_right_imag, color='#88c999', label='Z right')

        plt.xlabel('real-axis')
        plt.ylabel('imaginary-axis')
        plt.legend()
        plt.grid(True)
        plt.show()


def midpoint_3d(points1, points2):
    """
    Calculate the midpoint between two 3D points from a list of points made of the concatenation of left and right weak
    value vectors and scatter plot in 3D to check for variance (or scatter).
    """
    x_mid_points = []
    y_mid_points = []
    z_mid_points = []
    for c in range(len(points1)):
        mx = (points1[c][0] + points2[c][0]) / 2
        my = (points1[c][1] + points2[c][1]) / 2
        mz = (points1[c][2] + points2[c][2]) / 2

        x_mid_points.append(mx)
        y_mid_points.append(my)
        z_mid_points.append(mz)

    # creating figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Midpoints plot')
    ax.axis('scaled')

    ax.scatter(x_mid_points, y_mid_points, z_mid_points, color='b')

    plt.show()


def find_dist_around_ellipse(points1, points2, plot: bool = False):
    """
    Calculates the 3D distance between two "diametrically" opposed points (same index of location in each list) on the
    ellipse (formed by the left and right vectors)
    :param points1 - list of left weak value vector points (as 3 element list)
    :param points2 - list of right weak value vector points (as 3 element list)
    """

    def func(theta, amp, freq, phase_shift, offset):
        return amp * np.sin(freq * theta + phase_shift) + offset

    dist_list = []
    for c in range(len(points1)):
        p_left = points1[c]
        p_right = points2[c]
        dist = np.sqrt((p_left[0] - p_right[0]) ** 2 + (p_left[1] - p_right[1]) ** 2 + (p_left[2] - p_right[2]) ** 2)
        dist_list.append(dist)

    minor_axis = min(dist_list)
    major_axis = max(dist_list)
    index_major = dist_list.index(major_axis)
    index_minor = dist_list.index(minor_axis)

    semi_major_axis = np.around(major_axis / 2, decimals=5)
    semi_minor_axis = np.around(minor_axis / 2, decimals=5)

    max_left = points1[index_major]
    max_right = points2[index_major]
    min_left = points1[index_minor]
    min_right = points2[index_minor]
    center_min = [(min_left[c] + min_right[c]) / 2 for c in range(3)]
    center_max = [(max_left[c] + max_right[c]) / 2 for c in range(3)]
    # print(f"center_min: {center_min} | center_max: {center_max}")

    # find foci
    focal_length = np.around(np.sqrt(semi_major_axis ** 2 - semi_minor_axis ** 2), decimals=5)

    foci1 = [center_min[0] - focal_length, center_min[1], center_min[2]]
    foci2 = [center_min[0] + focal_length, center_min[1], center_min[2]]
    foci = [foci1, foci2]

    if plot:
        theta_data = np.linspace(-np.pi / 2, np.pi / 2, abs(index_major - index_minor))
        y_data = dist_list[index_minor: index_major] if index_minor < index_major else dist_list[
                                                                                       index_major: index_minor]
        popt, pcov = curve_fit(func, theta_data, y_data)

        x_points = [x for x in range(len(dist_list))]
        # Plot the generated points
        plt.figure()
        plt.scatter(x_points, dist_list, marker='o', s=10, label='data (weak values)')
        plt.scatter(x_points[index_minor: index_major], func(theta_data, *popt), marker='o', s=10,
                    label='fit: %5.3f * sin(%5.3f * theta + %5.3f) + %5.3f' % tuple(popt))
        plt.xlabel('position in weak value vector')
        plt.ylabel('3D distance')
        plt.title('Distance between "diametrically" opposed points on the ellipse')
        plt.grid(True)
        plt.legend()
        plt.show()

    data = {'max_left': max_left, 'max_right': max_right, 'min_left': min_left, 'min_right': min_right,
            'major_axis': semi_major_axis, 'minor_axis': semi_minor_axis, 'center': center_min, 'foci': foci,
            'focal_length': focal_length, 'index_major': index_major, 'index_minor': index_minor}

    return data


def fit_ellipse_wv(points1, points2, plot: bool = False):
    """
    Calculates the 3D distance between to "diametrically" opposed points (same index of location in each list) on the
    ellipse (formed by the left and right vectors)
    :param points1 - list of left weak value vector points (as 3 element list)
    :param points2 - list of right weak value vector points (as 3 element list)
    """

    def func(x, y, h, k, a, b):
        return ((x - h) / a) ** 2

    dist_list = []
    for c in range(len(points1)):
        p_left = points1[c]
        p_right = points2[c]
        dist = np.sqrt((p_left[0] - p_right[0]) ** 2 + (p_left[1] - p_right[1]) ** 2 + (p_left[2] - p_right[2]) ** 2)
        dist_list.append(dist)

    minor_axis = min(dist_list)
    major_axis = max(dist_list)
    index_major = dist_list.index(major_axis)
    index_minor = dist_list.index(minor_axis)

    max_left = points1[index_major]
    max_right = points2[index_major]
    min_left = points1[index_minor]
    min_right = points2[index_minor]
    center = [(np.array(min_left) + np.array(min_right)) / 2]

    print(major_axis / 2, minor_axis / 2)

    if plot:
        theta_data = np.linspace(-np.pi / 2, np.pi / 2, index_major - index_minor)
        y_data = dist_list[index_minor: index_major]
        popt, pcov = curve_fit(func, theta_data, y_data)

        x_points = [x for x in range(len(dist_list))]
        # Plot the generated points
        plt.figure()
        plt.scatter(x_points, dist_list, marker='o', s=10, label='data (weak values)')
        plt.scatter(x_points[index_minor: index_major], func(theta_data, *popt), marker='o', s=10,
                    label='fit: %5.3f * sin(%5.3f * theta + %5.3f) + %5.3f' % tuple(popt))
        plt.xlabel('position in weak value vector')
        plt.ylabel('3D distance')
        plt.title('Distance between "diametrically" opposed points on the ellipse')
        plt.grid(True)
        plt.legend()
        plt.show()

    return max_left, max_right, min_left, min_right, center


def ellipse_area(a, b):
    """
    Calculate the area of an ellipse.

    Parameters:
    - a: Semi-major axis length.
    - b: Semi-minor axis length.

    Returns:
    - Area of the ellipse.
    """
    area = np.pi * a * b
    return area


def eccentricity(a, b):
    """
    Calculate the eccentricity of an ellipse.

    Parameters:
    - a: Semi-major axis length.
    - b: Semi-minor axis length.

    Returns:
    - Eccentricity of the ellipse.
    """
    # Calculate the distance from the center to one of the foci (c)
    if a ** 2 - b ** 2 > 0:
        c = np.sqrt(a ** 2 - b ** 2)

        # Calculate the eccentricity (e)
        e = c / a
    else:
        e = np.nan

    return e


def angle_between_vectors(u, v):
    """
    Calculate the angle (in radians) between two vectors:
    u = vector generated by center and right most point on semi-major axis
    v = vector generated by center and start point on semi-major axis

    Parameters:
    center: (h, k) center coordinates
    max_right: coordinates of right most point on the semi-major axis
    start_point: starting point or first point of the ellipse (as it relates to the first weak value of the nth root of the swap)

    Returns:
    - Angle between the vectors in radians.
    """

    dot_product = np.dot(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

    # Avoid division by zero
    if magnitude_u == 0 or magnitude_v == 0:
        raise ValueError("Vectors should have non-zero magnitude.")

    cos_theta = dot_product / (magnitude_u * magnitude_v)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure the value is within valid range for arccos
    angle_deg = np.degrees(angle_rad)

    return angle_rad, angle_deg


def find_ellipse_axis(weak_values_all_left, weak_values_all_right):
    """
    Finds major and minor axis of an ellipse.
    First for loop goes through all possible distance combination to find the maximum distance and saves the 3D coordinates
    of the points and their index in the given list of points for the max distance (ellipse major-axis).
    Since it is now know that the maximum distance is at the diametrically opposed points, the second loop goes in the
    same direction for both lists to find the minimum distance (ellipse minor-axis).

    *** It was good initial validation but may deprecate since it's slow for large number of data points ***

    :param weak_values_all_left
    :param weak_values_all_right
    :return: minor_axis - minor axis
    :return: major_axis- major axis
    """
    dist_list = []
    for index_left, p_left in enumerate(weak_values_all_left):
        for index_right, p_right in enumerate(weak_values_all_right):
            dist = np.sqrt(
                (p_left[0] - p_right[0]) ** 2 + (p_left[1] - p_right[1]) ** 2 + (p_left[2] - p_right[2]) ** 2)
            dist_list.append(dist)

            if len(dist_list) > 1:
                if dist >= max(dist_list[:-1]):
                    max_left = (p_left, index_left)
                    max_right = (p_right, index_right)

    # print(f"Max point index left: {max_left[1]} | Max point index right {max_right[1]}")

    # find semi-minor axis and points
    dist_previous = max(dist_list)
    semi_minor_axis = 0
    for c in range(0, max_left[1] - 1, 1):

        p_left = weak_values_all_left[c]  # go in descending order (max_left[1] -> 0)
        p_right = weak_values_all_right[c]

        dist_temp = np.sqrt(
            (p_left[0] - p_right[0]) ** 2 + (p_left[1] - p_right[1]) ** 2 + (p_left[2] - p_right[2]) ** 2)
        if dist_temp <= dist_previous:
            semi_minor_axis = dist_temp
            dist_previous = dist_temp
            semi_minor_index = c

    min_left = weak_values_all_left[semi_minor_index]
    min_right = weak_values_all_right[semi_minor_index]

    # get center of ellipse
    mx = (min_left[0] + min_right[0]) / 2
    my = (min_left[1] + min_right[1]) / 2
    mz = (min_left[2] + min_right[2]) / 2
    center = [mx, my, mz]

    major_axis = max(dist_list) / 2
    minor_axis = semi_minor_axis / 2

    print(f"Ellipse Semi-Major Axis: {major_axis} (from find_ellipse_axis)")
    print(f"Ellipse Semi-Minor Axis: {minor_axis}")
    print(f"Points farthest apart on Semi-Major axis: {max_left[0]}, {max_right[0]}")
    print(f"Points farthest apart on Semi-Minor axis: {min_left}, {min_right}")

    return max_left, max_right, min_left, min_right, center


def ellipse_residuals(params, points):
    """
    Residuals function for the general ellipse equation to be minimized.

    Parameters:
    - params: Parameters of the ellipse (a, b, h, k, theta) where (h, k) is the center,
             and theta is the rotation angle.
    - points: Data points as a 2D array (shape: [2, num_points]).

    Returns:
    - Residuals between the actual data points and the points predicted by the ellipse equation.
    """
    a, b, h, k, theta = params

    # Rotate points back to the axis-aligned ellipse
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_points = np.array([
        cos_theta * (points[0] - h) + sin_theta * (points[1] - k),
        -sin_theta * (points[0] - h) + cos_theta * (points[1] - k)
    ])

    residuals = (
            (rotated_points[0] / a) ** 2 +
            (rotated_points[1] / b) ** 2 - 1
    )

    return np.sum(residuals ** 2)


def ellipse_constraints(params):
    """
    Constraints function enforcing a > 0, b > 0.

    Parameters:
    - params: Parameters of the ellipse (a, b, h, k).

    Returns:
    - Array of constraint violations.
    """
    a, b, _, _ = params
    return [a, b]


def fit_ellipse(x, y, major_axis=1.0, minor_axis=1.0):
    """
    Fit a general ellipse equation to a set of data points.

    Parameters:
    - x, y: Data points.

    Returns:
    - Tuple (a, b, h, k, theta) representing the parameters of the fitted ellipse equation.
    """
    # Stack points into a 2D array
    points = np.vstack((x, y))

    # Initial guess for the parameters (semi-major axis, semi-minor axis, center, angle)
    initial_guess = [major_axis, minor_axis, np.mean(x), np.mean(y), 0]

    # Use least squares optimization to fit the general ellipse equation to the data
    result = minimize(ellipse_residuals, initial_guess, args=(points,))

    # Extract the fitted parameters
    fitted_params = result.x
    a, b, h, k, theta = fitted_params

    return a, b, h, k, theta


# i is your initial state, f is your final state, S is the SpinOps or SpinOpsR function depending on whether you're finding weak
# values for the left or right wing, and U is your rotation about the x-axis or rSWAP matrix.
def evolve_state(i, f, U):
    """ Evolves states for i and f, forwards and backwards respectively.
  :param i - initial state (to be forward evolved)
  :param f - final state (to be backward evolved)
  :param U - gate to evolve the state through
  :return j vector of forward evolved state
  :return g - vector of backward evolved state
  """

    # j = U[0].dot(i)  # Applying U onto i as a forward transformation to obtain j
    # g = U[1].dot(f)  # Applying U inverse onto f as a backward transformation to obtain g

    j = np.matmul(U[0], i)
    g = np.matmul(U[1], f)

    # Check that norm of j (and g) is 1
    # print(f"Norm j: {np.linalg.norm(j)} | g: {np.linalg.norm(g)}")

    return j, g


def WeakValue(j, g, S, side: str = None):
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
    if side == "left":
        g_dot_j = np.around(np.inner(np.conj(g), j), decimals=7)
    elif side == "right":
        g_dot_j = np.around(np.inner(np.conj(g), j), decimals=7)
    else:
        raise ValueError("Must input a side! Either left of right")

    for w in range(3):  # To generate three components for weak values for each coordinate axis, x, y, and, z
        # Using np.conjugate and np.dot, we can code the formula for weak values. S[w] allows us to use each spin operator in
        # order through indices. S[0] is the x-component, for instance.
        try:
            W = np.inner(np.conj(g), S[w].dot(j)) / g_dot_j  # Weak value
            # W = np.inner(np.conj(S[w].dot(g)), j) / g_dot_j  # Weak value

        except ZeroDivisionError as e:
            print("Error: Cannot divide by zero")
            W = float('inf')
        Wreal.append(W.real)  # Appending the real part of W
        Wimag.append(W.imag)  # Appending the imaginary part of W

    return Wreal, Wimag, g_dot_j


def guess_n_hat(i, f, n, swap_iter, rot_step):
    """ Make a guess for n_hat """

    evo_operator = nestrs(n)

    evolved_j = [i]
    evolved_g = [f]

    # Generate forward and backward evolved states for Left and Right (3 left + 3 right 9 times (including initial states))
    j = np.matmul(evo_operator[0], evolved_j[-1])
    g = np.matmul(evo_operator[1], evolved_g[-1])

    evolved_j.append(j)
    evolved_g.append(g)

    iterations = swap_iter * (2 ** n)
    print(f"iterations: {iterations}")

    # 1) Run |i> forward (up to rotation point)
    for step in range(1, rot_step - 1):
        j = np.matmul(evo_operator[0], j)
        evolved_j.append(j)

    # 2) Run |f> backward (starting with a rotation
    for step in range(2, int(iterations)):
        g = np.matmul(evo_operator[1], g)
        evolved_g.append(g)

    # reverse time evolution of |f> so that it matches to |i> time evolution
    evolved_g = np.flip(evolved_g, axis=0)

    # print(f"length j: {len(evolved_j)}")
    # print(f"length g: {len(evolved_g)}")

    # 3) Calc the last weak value (Spinors) going forward in time
    # and return as guess for n_hat
    WVleft, WVleft_imag, g_dot_j_left = WeakValue(evolved_j[-1], evolved_g[len(evolved_j) - 1], SpinOps(), side="left")
    WVright, WVright_imag, g_dot_j_right = WeakValue(evolved_j[-1], evolved_g[len(evolved_j) - 1], SpinOpsR(),
                                                     side="right")

    print(f"Initial rotation: {WVleft}")

    return WVleft, WVleft_imag, WVright, WVright_imag


def calc_velocity(j, g, n=20):
    """ Calculate the velocity of the weak values.
    :param j - forward evolved state of |i>
    :param g - forward (not backward!) evolved state of |f>; by this point all g's have been flipped around such that
                they look forward evolved in time
    :param n - microgate root power
    """
    WVleft_before, WVleft_imag_before, _ = WeakValue(j, g, SpinOps(), side="left")
    WVright_before, WVright_imag_before, _ = WeakValue(j, g, SpinOpsR(), side="right")
    weak_vals_before = [WVleft_before, WVleft_imag_before, WVright_before, WVright_imag_before]

    # Send j and g through a root-swap micro-gate such that we can have an infinitesimal difference in the weak value
    # in order to calc the velocity (like the derivative definition)
    micro_gate = nestrs(n)
    j_micro = np.matmul(micro_gate[0], j)
    g_micro = np.matmul(micro_gate[0], g)

    WVleft_after, WVleft_imag_after, _ = WeakValue(j_micro, g_micro, SpinOps(), side="left")
    WVright_after, WVright_imag_after, _ = WeakValue(j_micro, g_micro, SpinOpsR(), side="right")
    weak_vals_after = [WVleft_after, WVleft_imag_after, WVright_after, WVright_imag_after]

    velocity_rot = np.array(
        [np.array(WVleft_after) - np.array(WVleft_before), np.array(WVleft_imag_after) - np.array(WVleft_imag_before),
         np.array(WVright_after) - np.array(WVright_before),
         np.array(WVright_imag_after) - np.array(WVright_imag_before)]) / (np.pi / (2 ** n))

    return velocity_rot, weak_vals_before, weak_vals_after


# Let's try n nth-root swap gates (equivalent to one root swap) and calculate Sx, Sy, Sz for each qubit (left + right)
def nroot_swap_weak_value_vectors(i, f, n,
                                  rot_step: int = 0,
                                  swap_iter: float = 1,
                                  one_qbit_rotation: bool = False,
                                  rotation_side: str = "real",
                                  n_hat=None):
    """
  Performs root-swap by performing n x 2 2^n-root swaps (forwards and backwards evolution of i and f, respectively)
  :param rot_step: step at which single gubit rotation will occur
  :param i - initial state to be forward evolved
  :param f - final state to be backward evolved
  :param n - power of n-root swap (1/2^n)
  :param rotation_side - determines whether the one qubit rotation is on the left or right
  :param swap_iter - determines number of swap iterations
  :param one_qbit_rotation - perform a one qubit rotation (at some random step, determined by rot_step)
  :param n_hat - guess for n_hat (rotation axis)
  :return weak_values_all_left - list of the left weak value vectors
  :return weak_values_all_right - list of the right weak value vectors
  :return table - table of values of the weak value vectors components
  """

    # Make a guess for n_hat if none given
    if n_hat is None:
        n_hat_left, n_hat_left_imag, n_hat_right, n_hat_right_imag = guess_n_hat(i=i, f=f, n=n,
                                                                                 swap_iter=swap_iter,
                                                                                 rot_step=rot_step)
        n_hat = n_hat_left if rotation_side == "real" else n_hat_left_imag

    evo_operator = nestrs(n)

    U, Uinv = Unitary(theta=np.pi / 200, n_hat=n_hat)

    evolved_j = [i]
    evolved_g = [f]
    WVleft_after_rot = None

    # Generate forward and backward evolved states for Left and Right (3 left + 3 right 9 times (including initial states))
    j, g = evolve_state(i, f, evo_operator)

    evolved_j.append(j)
    evolved_g.append(g)

    iterations = swap_iter * (2 ** n)
    # add one iteration step to account for missing points (for each side, left and right),
    # when adding a singe qubit gate
    if one_qbit_rotation:
        iterations -= 1
    # Randomly pick root swap step at which a unitary rotation about z will be performed
    # rot_step = rd.randint(5, iterations - 5)
    print(f"Rotation step: {rot_step}")

    # # TODO This only works for a middle rotation TODO
    # likely this evolution is wrong?? for the rotation and successive step the states are the same but the weak values aren't
    # for step in range(2, int(iterations)):
    #
    #     print(step)
    #     # Run one qubit rotation
    #     if one_qbit_rotation and (step == rot_step):
    #         print(f"j before: {j}")
    #         print(f"g before: {g}")
    #
    #         evolved_j.append(j)
    #         evolved_g.append(g)
    #
    #         print("Apply rotation")
    #         j, g = evolve_state(i=j, f=g, U=U)
    #         # do not store state j and g;
    #         # just evolve them - one extra step in the for loop is added to account for the missing point
    #         print(f"j after: {j}")
    #         print(f"g after: {g}")
    #
    #         evolved_j.append(j)
    #         evolved_g.append(g)
    #
    #         # WVleft_after_rot, _, _ = WeakValue(j=evolved_j[-1], g=g, S=SpinOps(), side="left")
    #
    #     j, g = evolve_state(j, g, evo_operator)
    #
    #     evolved_j.append(j)
    #     evolved_g.append(g)

    ####################################
    # Evolve |i> forwards before rotation
    ####################################
    for step in range(2, rot_step + 1):
        j = np.matmul(evo_operator[0], j)
        evolved_j.append(j)

    # Run one qubit rotation
    if one_qbit_rotation:
        j = np.matmul(U, j)
        evolved_j.append(j)

        # do not store state j;
        # just evolve it - one extra step in the for loop is added to account for the missing point

    # Continue evolution for |i> after rotation
    for step in range(rot_step + 1, int(iterations)):
        j = np.matmul(evo_operator[0], j)
        evolved_j.append(j)

    #################################################################
    # Evolve |f> backwards while applying the rotation gate correctly
    #################################################################
    for step in range(2, rot_step + 1):
        g = np.matmul(evo_operator[1], g)
        evolved_g.append(g)

    # Run one qubit rotation
    if one_qbit_rotation:
        g = np.matmul(Uinv, g)
        evolved_g.append(g)

    # Continue evolution for |f> after rotation
    for step in range(rot_step + 1, int(iterations)):
        g = np.matmul(evo_operator[1], g)
        evolved_g.append(g)

        # print(step)

    data = []

    # check that <f|i> is the same at every step (if not single qubit gate)
    f_dot_i_left = []
    f_dot_i_right = []
    l = len(evolved_j)

    weak_values_all_left = []  # list of left weak value vectors (list) for each step
    weak_values_all_right = []  # list of left weak value vectors (list) for each step
    weak_values_all_left_imag = []  # list of left weak value vectors (list) for each step
    weak_values_all_right_imag = []  # list of left weak value vectors (list) for each step

    # reverse time evolution of |f> so that it matches to |i> time evolution
    evolved_g = np.flip(evolved_g, axis=0)

    velocity_before_rot, _, weak_vals_before = calc_velocity(j=evolved_j[rot_step], g=evolved_g[rot_step])
    velocity_after_rot, _, weak_vals_after = calc_velocity(j=evolved_j[rot_step + 1], g=evolved_g[rot_step + 1])

    vel_mag_before = []
    for vec in velocity_before_rot:
        vel_mag_before.append(sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2))

    vel_mag_after = []
    for vec in velocity_after_rot:
        vel_mag_after.append(sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2))

    vel_percent_diff = []
    for c, val in enumerate(vel_mag_after):
        vel_percent_diff.append(abs(val - vel_mag_before[c]) / vel_mag_before[c])

    print(f"vel_percent_diff: {vel_percent_diff}")

    data_csv = [["Velocity before rotation (Re)"] + [str(x) for x in velocity_before_rot[:2][0]] + [' ', str(
        vel_mag_before[0])],
                ["Velocity after rotation (Re)"] + [str(x) for x in velocity_after_rot[:2][0]] + [' ', str(
                    vel_mag_after[0])] + [' ', vel_percent_diff[0]],
                ["Velocity before rotation (Im)"] + [str(x) for x in velocity_before_rot[:2][1]] + [' ', str(
                    vel_mag_before[1])],
                ["Velocity after rotation (Im)"] + [str(x) for x in velocity_after_rot[:2][1]] + [' ', str(
                    vel_mag_after[1])] + [' ', vel_percent_diff[1]]]

    # Check if change in the velocity obeys
    # Delta Re(velocity) = S2 x (Delta S1)
    # Delta Im(velocity) = W2 x (Delta S1)

    # delta_velocity = velocity_after_rot - velocity_before_rot
    # delta_weak_vals = np.array(weak_vals_after) - np.array(weak_vals_before)
    # # print(f"Re(delta_s1): {delta_s1}")
    # # print(f"Re(S2): {weak_vals_after[2]}") # Re(right)
    # cross_s2_ds1 = np.cross(weak_vals_after[2], delta_weak_vals[0])
    # # print(f"\nDelta Re(velocity) left: {delta_velocity[2]}")
    # # print(f"S2 x (Delta S1): {cross_s2_ds1}\n")

    for nr in range(0, l):

        WVleft, WVleft_imag, g_dot_j_left = WeakValue(evolved_j[nr], evolved_g[nr], SpinOps(), side="left")
        WVright, WVright_imag, g_dot_j_right = WeakValue(evolved_j[nr], evolved_g[nr], SpinOpsR(), side="right")

        # append <f|i>; g_dot_j_left and g_dot_j_right are the same since they use the same evolved_j and evolved_g
        f_dot_i_left.append(g_dot_j_left)
        f_dot_i_right.append(g_dot_j_right)
        weak_values_all_left.append(WVleft)
        weak_values_all_right.append(WVright)

        weak_values_all_left_imag.append(WVleft_imag)
        weak_values_all_right_imag.append(WVright_imag)

        datatemp = [nr + 1]
        for WV in [WVleft, WVright]:
            for x in range(3):
                datatemp.append(round(WV[x], 5))

        data.append(datatemp)

    # Calc weak values after rotation
    weak_vals_close = False
    if one_qbit_rotation:
        data_csv.append(["WV before rotation Left (Re)"] + [str(x) for x in weak_values_all_left[rot_step]])
        data_csv.append(["WV after rotation Left (Re)"] + [str(x) for x in weak_values_all_left[rot_step + 1]])

        data_csv.append(["WV before rotation Right (Re)"] + [str(x) for x in weak_values_all_right[rot_step]])
        data_csv.append(["WV after rotation Right (Re)"] + [str(x) for x in weak_values_all_right[rot_step + 1]])

        data_csv.append(["WV before rotation Left (Im)"] + [str(x) for x in weak_values_all_left_imag[rot_step]])
        data_csv.append(["WV after rotation Left (Im)"] + [str(x) for x in weak_values_all_left_imag[rot_step + 1]])

        data_csv.append(["WV before rotation Right (Im)"] + [str(x) for x in weak_values_all_right_imag[rot_step]])
        data_csv.append(["WV after rotation Right (Im)"] + [str(x) for x in weak_values_all_right_imag[rot_step + 1]])

        for row in data_csv:
            print(row)

        # print(f"WV before and after are exactly the same: {weak_values_all_left[rot_step] == weak_values_all_left[rot_step + 1]}")

        for c in range(3):
            weak_vals_close = cm.isclose(weak_values_all_left[rot_step][c], weak_values_all_left[rot_step + 1][c],
                                         rel_tol=.001)
            if not weak_vals_close:
                break

    table = ListTable(table_header=['n', 'Wlx', 'Wly', 'Wlz', 'Wrx', 'Wry', 'Wrz'], table_data=data)

    # round values
    weak_values_all_left = list(np.around(weak_values_all_left, decimals=4))
    weak_values_all_right = list(np.around(weak_values_all_right, decimals=4))
    weak_values_all_left_imag = list(np.around(weak_values_all_left_imag, decimals=4))
    weak_values_all_right_imag = list(np.around(weak_values_all_right_imag, decimals=4))

    i_qm_prediction = evolved_j[-1]

    # print(f"f_dot_i_left: {f_dot_i_left}")
    # print(f"f_dot_i_right: {f_dot_i_right}")
    # print(f"Average of <f|i> over all gate steps: {sum(f_dot_i_left)/len(f_dot_i_left)}")
    # print(f"< final f| initial i> {f_dot_i_left[0]} | < initial f| final i> {f_dot_i_left[-1]}")

    return weak_values_all_left, weak_values_all_right, weak_values_all_left_imag, weak_values_all_right_imag, \
           f_dot_i_left, one_qbit_rotation, weak_vals_close, data_csv, table.repr_html(), i_qm_prediction


def plot_ellipse(points, center, semi_major, semi_minor, foci, rotation_angle=0, num_points=100, plot=False):
    """
    Plot an ellipse given its center, semi-major and semi-minor axes, and rotation angle; and also plots a scatter
    of (x,y) points to visually check fit.

    Parameters:
    - points: weak value vector  points for scatter plot
    - center: Tuple (x, y) representing the center of the ellipse.
    - semi_major: Length of the semi-major axis.
    - semi_minor: Length of the semi-minor axis.
    - rotation_angle: Rotation angle of the ellipse in degrees (default is 0).
    - num_points: Number of points to use for plotting (default is 100).

    Returns:
    - None (plots the ellipse using matplotlib).
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    cos_theta = np.cos(np.radians(rotation_angle))
    sin_theta = np.sin(np.radians(rotation_angle))

    x = center[0] + semi_major * np.cos(theta) * cos_theta - semi_minor * np.sin(theta) * sin_theta
    y = center[1] + semi_major * np.cos(theta) * sin_theta + semi_minor * np.sin(theta) * cos_theta

    xy_points = []
    for c in range(len(x)):
        xy_points.append([x[c], y[c]])

    is_on_ellipse = check_points_on_ellipse(xy_points, center, semi_major_axis=semi_major, semi_minor_axis=semi_minor,
                                            foci=foci)

    if plot:
        x_weak_val, y_weak_val, _ = extract_xyz_coordinate(points)
        plt.plot(x, y, color='red', label='Ellipse')
        plt.plot(x_weak_val, y_weak_val, color='blue', label='Weak Values')
        plt.scatter(center[0], center[1], color='green', marker='o', label='Center')
        plt.title(f'Semi major and minor axis: {semi_major} | {semi_minor}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()

    return is_on_ellipse


# For testing only
def plot_html():
    # Helix equation
    t = np.linspace(0, 20, 100)
    x, y, z = np.cos(t), np.sin(t), t

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.write_html("templates/plot.html")  # Modify the html file
    fig.show()


def check(list):
    """
  Auxiliary function to check all elements in a list are the same:
  :param list - list of <f|i> dot products
  :return almost_same - boolean (depending on whether the numbers are within
                        the same relative tolerance - 10^-9 of one another)
  """
    for e in range(len(list)):
        almost_same = True if cm.isclose(list[0], list[e]) else False

    return almost_same


def check_conservation(WVleft, WVright):
    """
  Checks if the sum of the left and right weak value vectors (and magnitudes) are the same for all n-root SWAP steps
  :param WVleft - left weak value vector list
  :param WVright - right weak value vector list
  :return - boolean result of conservation check
  """
    magnitude_sums = []
    vector_sum = []
    for e in range(len(WVleft)):

        vector_sum_temp = []
        for w in range(3):
            vector_sum_temp.append(WVleft[e][w] + WVright[e][w])

        # save sum of left and right vector as list of tuples of x,y,z coordinates
        # for any later manipulation or computation
        vector_sum.append(tuple(vector_sum_temp))

        # calculate the magnitude of each sum (or each vector in vector_sum)
        sum_magnitude = 0
        for c in vector_sum_temp:
            sum_magnitude = sum_magnitude + c ** 2
        sum_magnitude = cm.sqrt(sum_magnitude)

        magnitude_sums.append(sum_magnitude)

    return check(magnitude_sums), vector_sum


def cross_prod(vec_left, vec_right, z_only=True):
    """
    Calculate cross product of two vectors (or left and right vectors)
    :param vec_left:
    :param vec_right:
    :param z_only: return only the z component of the cross product
    :return cross_prod:
    """
    cross_prod = []
    for _ in range(len(vec_left)):
        cross_vector = np.cross(vec_left, vec_right)
        magnitude = np.linalg.norm(cross_vector)
        if z_only:
            cross_prod.append((cross_vector / magnitude)[2])
        else:
            cross_prod.append(cross_vector / magnitude)

    return cross_prod


def calc_successive_angles(points_left, points_right, center, part, angle_choice="rad", rotation_step: int = None,
                           show_plot: bool = False):
    """
    Calculate the angles between consecutive vectors formed by subtracting the center point
    from each point in the given list of 3D points. Use these position vectors to also calculate cross product between left and right sides.

    Parameters:
    - points_list (list): A list of 3D points.
    - center (list): The center point for angle calculations.
    - part: real or imaginary string
    - angle_choice: radians or degrees
    - rotation_step: step at which rotation occurs to split the left and right point to before and after
    - show_plot: boolean - whether to display plot

    Returns:
    - angles (list): A list of angles in degrees between consecutive vectors.
    """

    # # creating figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # title = ax.set_title('3D Plot')
    # ax.axis('scaled')

    points_list = points_left + points_right
    if len(points_list) < 2:
        raise ValueError("At least two points are required to calculate angles.")

    angles = []
    pos_vec_all = []  # left and right position vectors
    diff_vec_all = []  # difference between successive vectors for left and right vectors

    if rotation_step is not None:
        points = [points_left[:rotation_step - 1], points_left[rotation_step:],
                  points_right[:rotation_step - 1], points_right[rotation_step:]]
    else:
        points = [points_left, points_right]

    for vec in points:
        pos_vec = []
        diff_vec = []
        for i in range(len(vec) - 1):
            vector1 = np.array(vec[i]) - np.array(center)
            vector2 = np.array(vec[i + 1]) - np.array(center)
            pos_vec.append(list(vector1))
            diff_vec.append(list(vector2 - vector1))

            dot_product = np.dot(vector1, vector2)
            magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

            if magnitude_product == 0:
                raise ValueError("Vectors cannot have zero magnitude.")

            angle_rad = np.arccos(dot_product / magnitude_product)
            angle_deg = np.degrees(angle_rad)
            if angle_choice == "deg":
                angles.append(angle_deg)
            else:
                angles.append(angle_rad)

        pos_vec_all.append(pos_vec)
        diff_vec_all.append(diff_vec)

    if rotation_step is not None:
        velocity_points = {part: {"left": {"before": diff_vec_all[0],
                                           "after": diff_vec_all[1]},
                                  "right": {"before": diff_vec_all[2],
                                            "after": diff_vec_all[3]}
                                  }
                           }

        spin_vec_points = {part: {"left": {"before": pos_vec_all[0],
                                           "after": pos_vec_all[1]},
                                  "right": {"before": pos_vec_all[2],
                                            "after": pos_vec_all[3]}
                                  }
                           }
    else:
        velocity_points = {}
        spin_vec_points = {}

    print(
        f"Sum of all step angles{'' if angle_choice == 'deg' else ' (multiple of pi)'}: {sum(angles) if angle_choice == 'deg' else sum(angles) / np.pi}")

    # TODO check why cross_prod function is not working (not returning the z component only and not returning a normalized vec)
    #  to replace the 2 for loops
    # Calculate cross product between left and right sides (spin vectors)
    cross_prod_rl_spinors = []
    cross_prod_rl_diff = []

    if rotation_step is None:
        for i, vec_left in enumerate(pos_vec_all[0]):
            cross_vector = np.cross(vec_left, pos_vec_all[1][i])
            magnitude = np.linalg.norm(cross_vector)
            cross_prod_rl_spinors.append((cross_vector / magnitude)[2])

        # cross_prod_rl_diff = cross_prod(vec_left=diff_vec_all[0], vec_right=diff_vec_all[1])
        for i, diff_left in enumerate(diff_vec_all[0]):
            cross_vector = np.cross(diff_left, diff_vec_all[1][i])
            magnitude_diff = np.linalg.norm(cross_vector)

            # TODO account for zero magnitude cross product
            # if magnitude_diff == 0:
            #     raise ValueError(f"Vectors cannot have zero magnitude. (Magnitude: {magnitude_diff} | "
            #                      f"Cross product vector: {cross_vector} | Vectors left: {diff_left}, right: {diff_vec_all[1][i]})")

            cross_prod_rl_diff.append((cross_vector / magnitude_diff)[2])

    x_data = np.linspace(0, len(points_list) - 1, num=len(points_list) - 2)
    x_data_cross = np.linspace(0, len(points_list) - 1, num=len(points_list) - 2)

    if show_plot:
        plt.scatter(x_data, angles, color='blue' if part == "real" else "red", label=f'Angular velocity ({part})')
        # plt.scatter(x_data_cross, cross_prod, color='green' if part == "black" else "red", label=f'Angular velocity ({part})')

        plt.xlabel('Gate step')
        plt.ylabel(f'Angle [{angle_choice}]')
        plt.title('Angular velocity at every gate step (with constant time step = 1)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return angles, velocity_points, spin_vec_points, cross_prod_rl_spinors, cross_prod_rl_diff


def get_input_states(i_state: str = "separable", f_state: str = "separable"):
    if i_state == "separable":  # separable
        i = sepstate()
    elif i_state == "entangled":  # entangled
        i = entstate()
    else:
        i = None  # None is user defined state

    if f_state == "separable":  # separable
        f = sepstate()
    elif f_state == "entangled":  # entangled
        f = entstate()
    else:
        f = None  # None is user defined state

    return i, f


def run_weak_vals(num_runs: int = 1, rot_side: str = "real"):
    """
    Run weak values app.
    :param rot_side: determines whether the one qubit rotation will be on the left or right
    :param num_runs: number of iterations to run
    :return:
    """
    ellipse_dict_all = []
    time_start = time.time()
    success = []
    a_val = []
    csv_data = []
    run = 0

    while run < num_runs:
        run += 1
        print(f"Run: {run}")

        i, f = get_input_states()

        n = 6
        rot_step = int(2 ** n / 2) - 1

        init_state = f"Initial state: {i}"
        final_state = f"Final state: {f}"
        print("\n" + init_state)
        print(final_state)
        csv_data += [[" "],
                     [f"Run {run} | Power of root swap {n}"],
                     [init_state],
                     [final_state],
                     [' ', 'X', 'Y', 'Z', ' ', 'Velocity Magnitude', ' ', 'Velocity Percent Diff']]

        # First iteration of weak values and second guess
        data_csv = [" "]
        (weak_values_all_left_real,
         weak_values_all_right_real,
         weak_values_all_left_imag,
         weak_values_all_right_imag,
         f_dot_i_left,
         single_qbit_rotation,
         weak_vals_close,
         data_csv, _, _) = nroot_swap_weak_value_vectors(i=i, f=f, n=n,
                                                         swap_iter=1.0,
                                                         rot_step=rot_step,
                                                         # rotation_side=rotation_side, TODO
                                                         one_qbit_rotation=True)

        print(f"weak_vals_close: {weak_vals_close}")
        if single_qbit_rotation:
            trials = 1
            max_trials = 10

            while not weak_vals_close and trials < max_trials:
                print(f"Rotation axis: {weak_values_all_left_real[rot_step]}")

                n_hat = weak_values_all_left_real[rot_step] if rot_side == "real" else weak_values_all_left_imag[rot_step]
                (weak_values_all_left_real,
                 weak_values_all_right_real,
                 weak_values_all_left_imag,
                 weak_values_all_right_imag,
                 f_dot_i_left,
                 single_qbit_rotation,
                 weak_vals_close,
                 data_csv, _, _) = nroot_swap_weak_value_vectors(i=i, f=f, n=n,
                                                                 swap_iter=1.0,
                                                                 rot_step=rot_step,
                                                                 # rotation_side=rotation_side, TODO
                                                                 one_qbit_rotation=single_qbit_rotation,
                                                                 n_hat=n_hat)

                trials += 1

            if trials < max_trials:
                print(f"Iteration converged after {trials} steps")
                csv_data += data_csv
            else:
                print("Iteration didn't converge!")
                csv_data += [["Iteration didn't converge!"]]

                num_runs += 1
                print(f"num_runs: {num_runs}")

            print(f"data_csv: {data_csv}")

            # csv_data.append([" "])

        ellipse_main = {"i": i, "f": f, "root_power": n}

        print(f"Check if <f|i> is the same at every step of the n-root SWAP: {check(f_dot_i_left)}")

        # split weak values into 2 sets if introducing a rotation
        # if single_qbit_rotation:
        #     before_rotation = [weak_values_all_left_real[:rot_step], weak_values_all_right_real[:rot_step], weak_values_all_left_imag[:rot_step], weak_values_all_right_imag[:rot_step]]
        #     after_rotation = [weak_values_all_left_real[rot_step:], weak_values_all_right_real[rot_step:], weak_values_all_left_imag[rot_step:], weak_values_all_right_imag[rot_step:]]

        # for c, weak_vals in enumerate([before_rotation, after_rotation]):
        #
        #     rot_msg = 'BEFORE' if c == 0 else 'AFTER'
        #     print(rot_msg + " rotation")

        # weak_values_all_left_real, weak_values_all_right_real, weak_values_all_left_imag, weak_values_all_right_imag = weak_vals

        plot_3d_radius(weak_values_left=weak_values_all_left_real, weak_values_right=weak_values_all_right_real,
                       weak_values_left_imag=weak_values_all_left_imag,
                       weak_values_right_imag=weak_values_all_right_imag,
                       show_plot=False)

        plot_complex_plane(weak_values_left=weak_values_all_left_real, weak_values_right=weak_values_all_right_real,
                           weak_values_left_imag=weak_values_all_left_imag,
                           weak_values_right_imag=weak_values_all_right_imag, show_plot=False)

        if not single_qbit_rotation:
            plot_axis_coordinate(weak_values_left=weak_values_all_left_real,
                                 weak_values_right=weak_values_all_right_real,
                                 weak_values_left_imag=weak_values_all_left_imag,
                                 weak_values_right_imag=weak_values_all_right_imag, show_plot=False)

        magnitude_conservation, vector_sum = check_conservation(weak_values_all_left_real, weak_values_all_right_real)
        print(
            f"Checks if the sum of the left and right weak value vectors magnitudes are the same for all n-root SWAP steps: {magnitude_conservation}")

        # final rotated values that may still have an angle or rotation for ellipse that is not accounted for
        _, _, final_rotated_vals, plot_dict, _ = plot_weak_values(weak_values_all_left_real, weak_values_all_right_real,
                                                                  weak_values_all_left_imag, weak_values_all_right_imag,
                                                                  plot_quiver=False, plot_plane=False, show_plot=False)
        key_list = list(final_rotated_vals.keys())
        for c in range(len(key_list)):

            part = key_list[c]  # 'real' or 'imaginary'
            print(f"\nRunning ellipse check for {part} weak values:")

            # Get weak values for real or imaginary part
            weak_values_all_left_rotated = final_rotated_vals[part][0]
            weak_values_all_right_rotated = final_rotated_vals[part][1]

            _, velocity_points, spin_vec_points, _, _ = calc_successive_angles(points_left=weak_values_all_left_rotated,
                                                                               points_right=weak_values_all_right_rotated,
                                                                               part=part,
                                                                               center=[0, 0, 0], rotation_step=rot_step,
                                                                               show_plot=False)
            print(f"velocity: {velocity_points}")
            print(f"spin vecs: {spin_vec_points}")

            ellipse_data_rot = find_dist_around_ellipse(weak_values_all_left_rotated, weak_values_all_right_rotated)
            # print("\nCheck points on ellipse with calculated values of a, b, h, k (not from fit) and generated values of x,y points:")
            # plot_ellipse(points=weak_values_all_left_rotated + weak_values_all_right_rotated, foci=foci, center=center, semi_major=major_axis, semi_minor=minor_axis)

            major_axis = ellipse_data_rot['major_axis']
            minor_axis = ellipse_data_rot['minor_axis']
            focal_length = ellipse_data_rot['focal_length']
            index_major = ellipse_data_rot['index_major']

            xy_left = extract_xyz_coordinate(weak_values_all_left_rotated)
            xy_right = extract_xyz_coordinate(weak_values_all_right_rotated)

            a = 100  # default large value for semi-major axis
            # re-run parameter fit if Semi-Major Axis (a) is too large
            cond = True
            stop = 0
            while cond and stop <= 100:
                a, b, h, k, theta = fit_ellipse(xy_left[0] + xy_right[0], xy_left[1] + xy_right[1],
                                                major_axis=major_axis,
                                                minor_axis=ellipse_data_rot['minor_axis'])
                cond = not (cm.isclose(a, major_axis, rel_tol=1))
                stop += 1

            print("Semi-Major Axis (a):", a)
            print("Semi-Minor Axis (b):", b)
            print("Center (h, k):", h, k)
            print(f"Angle of rotation: {np.degrees(theta)}")

            # print("\nCheck points on ellipse with fit (minimizing function) values of a, b, h, k (not including theta) and weak values of x,y points:")
            # plot_ellipse(points=weak_values_all_left_rotated + weak_values_all_right_rotated, foci=foci, center=[h, k, 0], semi_major=a, semi_minor=b)
            print(f"\nRechecking points on ellipse for weak values with transformation angle {np.degrees(theta)}:")
            # apply angle transformation
            xy_rot_by_theta_left = []
            for point in weak_values_all_left_rotated:
                x_trans = (point[0] - h) * np.cos(theta) + (point[1] - k) * np.sin(theta)
                y_trans = -(point[0] - h) * np.sin(theta) + (point[1] - k) * np.cos(theta)
                xy_rot_by_theta_left.append([x_trans, y_trans, 0])

            xy_rot_by_theta_right = []
            for point in weak_values_all_right_rotated:
                x_trans = (point[0] - h) * np.cos(theta) + (point[1] - k) * np.sin(theta)
                y_trans = -(point[0] - h) * np.sin(theta) + (point[1] - k) * np.cos(theta)
                xy_rot_by_theta_right.append([x_trans, y_trans, 0])

            # refit with new x,y values
            xy_rot_by_theta = xy_rot_by_theta_left + xy_rot_by_theta_right
            xy = extract_xyz_coordinate(xy_rot_by_theta)
            cond = True
            stop = 0
            while cond and stop <= 100:
                a, b, h, k, theta = fit_ellipse(xy[0], xy[1], major_axis=major_axis, minor_axis=minor_axis)
                cond = not (cm.isclose(a, major_axis, rel_tol=1))
                stop += 1

            print("Semi-Major Axis (a):", a)
            print("Semi-Minor Axis (b):", b)
            print("Center (h, k):", h, k)
            print(f"Angle of rotation: {np.degrees(theta)}")
            print(f"Focal length: {focal_length}")
            print(f"Eccentricity: {eccentricity(a, b)}")

            is_on_ellipse = plot_ellipse(points=xy_rot_by_theta, foci=ellipse_data_rot['foci'], center=[h, k, 0],
                                         semi_major=a, semi_minor=b)
            success.append(is_on_ellipse)
            if not success[-1]:
                a_val.append([a, major_axis])
                ellipse_main[f"{part}"] = {}
            else:
                _, _, _, spinor_cross_prod, diff_cross_prod = calc_successive_angles(points_left=xy_rot_by_theta_left,
                                                                                     points_right=xy_rot_by_theta_right,
                                                                                     part=part, center=[h, k, 0],
                                                                                     show_plot=False)

                # update dictionary if success
                ellipse_basic = {}
                subscript = '_r' if part == 'real' else '_i'
                ellipse_basic["a" + subscript] = a
                ellipse_basic["b" + subscript] = b
                # ellipse_basic["center" + subscript] = [h, k]
                ellipse_basic["focal_length" + subscript] = focal_length
                ellipse_basic["area" + subscript] = np.pi * a * b
                ellipse_basic["eccentricity" + subscript] = eccentricity(a, b)
                ellipse_basic["hits_major_axis" + subscript] = index_major
                ellipse_basic["root_swap_fraction" + subscript] = 1 - index_major / len(weak_values_all_left_rotated)
                ellipse_basic["spinor_cross_prod" + subscript] = spinor_cross_prod[:10]
                ellipse_basic["diff_cross_prod" + subscript] = diff_cross_prod[:10]

                # Calculate angle of where ellipse starts with respect to the major axis on the right
                start_point = plot_dict[part]['start_point_left']
                ellipse_basic["start_angle_left" + subscript] = start_point
                u = np.array([ellipse_data_rot['max_right'][0] - h, ellipse_data_rot['max_right'][1] - k])
                v = np.array([start_point[0] - h, start_point[1] - k])

                # Calculate start angle between right most point on major axis and starting point (picked from the left
                # half of the ellipse)
                ellipse_basic["start_angle_left" + subscript] = angle_between_vectors(u, v)[1][0]

                # Calculate the cross product of 2 vectors on the left side (or right) to get the direction of the ellipse
                ellipse_basic['direction_left' + subscript] = calc_normal_vector(p1=[h, k],
                                                                                 p2=weak_values_all_left_rotated[0],
                                                                                 p3=weak_values_all_left_rotated[5])
                ellipse_basic['direction_right' + subscript] = calc_normal_vector(p1=[h, k],
                                                                                  p2=weak_values_all_right_rotated[0],
                                                                                  p3=weak_values_all_right_rotated[5])

                ellipse_basic["normal_vec" + subscript] = plot_dict[part]["normal_vec"]
                ellipse_main[f"{part}"] = ellipse_basic
                ellipse_main[f"{part}"][f"velocity_{part}"] = velocity_points[part]
                ellipse_main[f"{part}"][f"spin_vecs_{part}"] = spin_vec_points[part]

                # Calculate angle between normal and center to center vector
                ellipse_data_r = find_dist_around_ellipse(weak_values_all_left_real, weak_values_all_right_real)
                ellipse_data_i = find_dist_around_ellipse(weak_values_all_left_imag, weak_values_all_right_imag)

                center_r = ellipse_data_r['center']
                center_i = ellipse_data_i['center']
                center_to_center_vec_i = [center_r[c] - center_i[c] for c in range(3)]
                center_to_center_vec_r = [center_i[c] - center_r[c] for c in range(3)]

                # TODO fix so that real and imaginary keys don't throw an error if running with either real or imaginary parts
                # ellipse_main["real"]['angle_normal_r_ctc'] = angle_between_vectors(center_to_center_vec_i,
                #                                                                    ellipse_main["real"]['normal_vec_r'])[1]  # ctc - short for center to center vector
                # ellipse_main['imaginary']['angle_normal_i_ctc'] = angle_between_vectors(center_to_center_vec_r, ellipse_main['imaginary']['normal_vec_i'])[1]
                #
                # # Calculate angle between normals
                # normal_real = ellipse_main['real']['normal_vec_r']
                # normal_imag = ellipse_main['imaginary']['normal_vec_i']
                # ellipse_main['angle_between_normals'] = angle_between_vectors(normal_real, normal_imag)[1]
                #
                # # Calculate angles between ellipse normal and center to origin vector
                # nr = ellipse_main['real']['normal_vec_r']
                # ellipse_main['real']['angle_normal_otc_r'] = angle_between_vectors(nr, np.array(center_r))[1]  # otc - short for origin to center vector
                # ni = ellipse_main['imaginary']['normal_vec_i']
                # ellipse_main['imaginary']['angle_normal_otc_i'] = angle_between_vectors(nr, np.array(center_i))[1]  # otc - short for origin to center vector

                # Calculate angle between ellipse normal and triangle normal
                # triangle_normal = calc_normal_vector(p1=[0, 0, 0], p2=center_r, p3=center_i)
                # ellipse_main['real']['angle_normal_triangle_r'] = angle_between_vectors(triangle_normal, normal_real)[1]
                # ellipse_main['imaginary']['angle_normal_triangle_i'] = angle_between_vectors(triangle_normal, normal_imag)[1]

                ellipse_dict_all.append(ellipse_main)

        del final_rotated_vals, weak_values_all_left_rotated, weak_values_all_right_rotated, xy_left, xy_right, \
            xy_rot_by_theta, xy, weak_values_all_left_real, weak_values_all_right_real, weak_values_all_left_imag, \
            weak_values_all_right_imag
        gc.collect()

    if all(success):
        print("All succeeded!")
    else:
        print(len(a_val), a_val)

    # Creating histogram
    # fig, ax = plt.subplots()
    # ax.hist(np.array(hist_area_real), bins=40, label='area real') #np.linspace(int(min(a)), int(max(a)), 1)
    # ax.hist(np.array(angle_between_normals), bins=40, label='angle between ellipse normals', alpha=.5)
    #
    # plt.legend()
    # plt.show()

    # field names
    fields = list(ellipse_dict_all[0]['real'].keys()) + list(ellipse_dict_all[0]['imaginary'].keys())

    # name of csv file
    filename = "ellipse_data.csv"

    # Round numbers
    round_num = False
    if round_num:
        for ellipse in ellipse_dict_all:
            for part in ['real', 'imaginary']:
                for key in ellipse[part].keys():
                    ellipse[part][key] = np.around(ellipse[part][key], decimals=4)

    # writing to csv file
    # with open(filename, 'w') as csvfile:
    #     # creating a csv dict writer object
    #     writer = csv.DictWriter(csvfile, fieldnames=fields)
    #
    #     # writing headers (field names)
    #     writer.writeheader()
    #
    #     # writing data rows
    #     for ellipse in ellipse_dict_all:
    #         print(ellipse["real"])
    #         print(ellipse["imaginary"])
    #
    #         if (ellipse["real"] is not None) or (ellipse["imaginary"] is not None):
    #             writer.writerows([ellipse["real"]] + [ellipse["imaginary"]])

    time_stop = time.time()
    run_time = time_stop - time_start
    print(f"Total run time: {run_time}")
    print(f"Time per single run (average): {run_time / num_runs}")

    # name of csv file
    filename = "velocity_with_rotation.csv"

    print(csv_data)

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the data rows
        csvwriter.writerows(csv_data)


if __name__ == "__main__":
    num_runs = int(input("\n Please enter the number of runs: "))  # number of states to iterate
    run_weak_vals(num_runs=num_runs)

