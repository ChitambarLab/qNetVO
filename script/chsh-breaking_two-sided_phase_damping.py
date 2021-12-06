from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from datetime import datetime
import matplotlib.pyplot as plt
import json
from context import QNetOptimizer as QNopt

"""
This script collects data about the noise robustness of the CHSH Bell inequality
with respect to a phase damping channel acting each side of the entangled
system.
"""


def Rot_meas_ansatz(settings, wires):
    qml.Rot(*settings[0:3], wires=wires[0])


def chsh_noise_optimization(prep_nodes, meas_nodes, **opt_kwargs):
    def _optimization_fn(arg1, arg2):
        print("noise aregs ares ", arg1, ", ", arg2)
        noise_nodes = [
            QNopt.NoiseNode(
                [0], lambda settings, wires: qml.PhaseDamping(arg1, wires=wires[0])
            ),
            QNopt.NoiseNode(
                [1], lambda settings, wires: qml.PhaseDamping(arg2, wires=wires[0])
            )
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, noise_nodes)
        chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz)
        init_settings = chsh_ansatz.rand_scenario_settings()

        try:
            opt_dict = QNopt.gradient_descent(chsh_cost, init_settings, **opt_kwargs)
        except Exception as err:
            print("An error occurred during gradient descent.")
            print(err)
            opt_dict = {
                "opt_score": np.nan,
                "opt_settings": [[], []],
                "scores": [np.nan],
                "samples": [0],
                "settings_history": [[[], []]],
            }

        print("noise params : ", arg1, ", ", arg2)
        print("max score : ", opt_dict["opt_score"])

        return opt_dict

    return _optimization_fn


def save_optimization_results(ansatz_name, x_range, y_range, opt_dicts):
    json_data = {"x_mesh" : [[]], "y_mesh" : [[]], "max_scores": [], "opt_settings": []}

    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    json_data["x_mesh"] = x_mesh.tolist()
    json_data["y_mesh"] = y_mesh.tolist()

    for row_id in range(x_mesh.shape[0]):
        json_data["max_scores"].append([])
        json_data["opt_settings"].append([])
        for col_id in range(x_mesh.shape[1]):
            opt_id = col_id*x_mesh.shape[0] + row_id

            max_score = max(opt_dicts[opt_id]["scores"])
            max_id = opt_dicts[opt_id]["scores"].index(max_score)
            max_sample = opt_dicts[opt_id]["samples"][max_id]
            opt_settings = opt_dicts[opt_id]["settings_history"][max_id]

            json_data["max_scores"][row_id] += [max_score]
            json_data["opt_settings"][row_id] += [QNopt.settings_to_list(opt_settings)]

            plt.plot(
            opt_dicts[opt_id]["samples"],
            opt_dicts[opt_id]["scores"], "--.")
            plt.plot([max_sample], [max_score], "r*")


    print(json_data["max_scores"])

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    filename = "script/data/chsh-breaking_two-sided_phase_damping/" + ansatz_name + "/" + datetime_ext
    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_data, indent=2))

    plt.plot(
        opt_dicts[0]["samples"],
        [2 * np.sqrt(2)] * len(opt_dicts[0]["samples"]),
        label="Quantum Bound",
    )
    plt.plot(opt_dicts[0]["samples"], [2] * len(opt_dicts[0]["samples"]), label="Classical Bound")
    plt.title("Two-Sided Phase Damping in CHSH Scenario:\n" + ansatz_name)
    plt.ylabel("CHSH Score")
    plt.xlabel("Epoch")
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":

    client = Client(processes=True)

    max_entangled_prep = [
        QNopt.PrepareNode(1, [0, 1], QNopt.max_entangled_state, 3),
    ]
    arb_prep = [QNopt.PrepareNode(1, [0, 1], qml.templates.subroutines.ArbitraryUnitary, 15)]
    meas_nodes = [
        QNopt.MeasureNode(2, 2, [0], Rot_meas_ansatz, 3),
        QNopt.MeasureNode(2, 2, [1], Rot_meas_ansatz, 3),
    ]

    x_range = np.arange(0, 1.01, 1/10)
    y_range = np.arange(0, 1.01, 1/10)

    params_range = np.zeros((2,len(x_range)*len(y_range)))
    for x_id in range(len(x_range)):
        for y_id in range(len(y_range)):
            params_range[:, x_id*len(y_range) + y_id] = [x_range[x_id], y_range[y_id]]

    max_entangled_optimization = chsh_noise_optimization(
        max_entangled_prep,
        meas_nodes,
        sample_width=5,
        step_size=0.15,
        num_steps=40,
        verbose=False
    )
    max_entangled_jobs = client.map(max_entangled_optimization, *params_range)
    max_entangled_opts = client.gather(max_entangled_jobs)
    save_optimization_results("max_entangled", x_range, y_range, max_entangled_opts)

    client.restart()

    arb_optimization = chsh_noise_optimization(
        arb_prep, meas_nodes,
        sample_width=5,
        step_size=0.15,
        num_steps=40,
        verbose=False
    )

    params_range = np.zeros((2,len(x_range)*len(y_range)))
    for x_id in range(len(x_range)):
        for y_id in range(len(y_range)):
            params_range[:, x_id*len(y_range) + y_id] = [x_range[x_id], y_range[y_id]]

    arb_jobs = client.map(arb_optimization, *params_range)
    arb_opts = client.gather(arb_jobs)
    save_optimization_results("arbitrary", x_range, y_range, arb_opts)
