from flask import Flask, render_template, request
from weak_values import get_input_states, plot_weak_values, nroot_swap_weak_value_vectors
from waitress import serve
import os

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/weak_values/')
def run_app():
    runs = request.args.get('runs')
    if runs is None or not bool(runs.strip()):
        runs = "10"
    runs = int(runs)

    power = request.args.get('power')
    if power is None or not bool(power.strip()):
        power = "6"
    power = int(power)

    swap_iter = request.args.get('swap_iter')
    if swap_iter is None or not bool(swap_iter.strip()):
        swap_iter = "1"
    swap_iter = float(swap_iter)

    agree_i = request.args.get('agree_i', 'separable')
    agree_f = request.args.get('agree_f', 'separable')

    show_plot = request.args.get('show_plot') == "yes"
    single_gate_rotation = request.args.get('single_gate_rotation') == "yes"

    rot_step = 0
    rotation_side = "real"
    if single_gate_rotation:
        rot_step = request.args.get('rot_step', int(2 ** (power - 1)))
        rotation_side = request.args.get('rotation_side')

    i, f = get_input_states(i_state=agree_i, f_state=agree_f)
    if i is None:
        i = request.args.get('i')
    if f is None:
        f = request.args.get('f')

    (weak_values_all_left_real, weak_values_all_right_real, weak_values_all_left_imag, weak_values_all_right_imag,
     f_dot_i_left, single_qbit_rotation, weak_vals_close, data_csv, i_qm_prediction) = nroot_swap_weak_value_vectors(
        i=i, f=f, n=power, swap_iter=swap_iter, rot_step=rot_step, rotation_side=rotation_side,
        one_qbit_rotation=single_gate_rotation)

    _, _, final_rotated_vals, plot_dict, _ = plot_weak_values(weak_values_all_left=weak_values_all_left_real,
                                                              weak_values_all_right=weak_values_all_right_real,
                                                              weak_values_all_left_imag=weak_values_all_left_imag,
                                                              weak_values_all_right_imag=weak_values_all_right_imag,
                                                              show_plot=show_plot)

    return render_template("weak_values.html", title="Weak Values Web App", i=f"{i}", f=f"{f}",
                           start_left_real=f"{weak_values_all_left_real[0]}",
                           finish_left_real=f"{weak_values_all_left_real[-1]}",
                           start_right_real=f"{weak_values_all_right_real[0]}",
                           finish_right_real=f"{weak_values_all_right_real[-1]}",
                           start_left_imag=f"{weak_values_all_left_imag[0]}",
                           finish_left_imag=f"{weak_values_all_left_imag[-1]}",
                           start_right_imag=f"{weak_values_all_right_imag[0]}",
                           finish_right_imag=f"{weak_values_all_right_imag[-1]}",
                           i_qm_predict=f"{i_qm_prediction}",
                           single_gate_point_real_left=f"{weak_values_all_left_real[rot_step] if single_gate_rotation else 'No rotation'}",
                           single_gate_point_real_right=f"{weak_values_all_right_real[rot_step] if single_gate_rotation else 'No rotation'}",
                           single_gate_point_imag_left=f"{weak_values_all_left_imag[rot_step] if single_gate_rotation else 'No rotation'}",
                           single_gate_point_imag_right=f"{weak_values_all_right_imag[rot_step] if single_gate_rotation else 'No rotation'}")

@app.route('/plot/')
def plot_view():
    plot_file = open(os.getcwd() + "/weak_values_GUI/plot.html", "r+", encoding="utf-8")
    return ''.join(plot_file.readlines())

@app.route('/results/')
def results_view():
    results = open("results.html", "r+", encoding="utf-8")
    return ''.join(results.readlines())

@app.route('/about')
def about():
    return 'The about page'

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:8000")
    try:
        serve(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"Error starting server: {e}")