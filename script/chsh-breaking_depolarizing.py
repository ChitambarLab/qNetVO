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
with respect to an depolarizing channel acting on one side of the entangled
system.
"""


def Rot_CNOT_Rot_prep(settings, wires):
    qml.Rot(*settings[0:3], wires=wires[0])
    qml.Rot(*settings[3:6], wires=wires[1])
    qml.CNOT(wires=wires[0:2])
    qml.Rot(*settings[6:9], wires=wires[0])
    qml.Rot(*settings[9:12], wires=wires[1])


def Rot_meas_ansatz(settings, wires):
    qml.Rot(*settings[0:3], wires=wires[0])


def chsh_noise_optimization(prep_nodes, meas_nodes, **opt_kwargs):
    def _optimization_fn(noise_args):
        noise_nodes = [
            QNopt.NoiseNode(
                [1], lambda settings, wires: qml.DepolarizingChannel(noise_args, wires=wires[0])
            ),
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

        print("noise params : ", noise_args)
        print("max score : ", opt_dict["opt_score"])

        return opt_dict

    return _optimization_fn


def save_optimization_results(ansatz_name, param_range, opt_dicts):
    json_data = {"noise_params": [], "max_scores": [], "opt_settings": []}

    for i in range(len(param_range)):
        noise_param = float(param_range[i])
        json_data["noise_params"] += [noise_param]

        max_score = max(opt_dicts[i]["scores"])
        max_id = opt_dicts[i]["scores"].index(max_score)
        max_sample = opt_dicts[i]["samples"][max_id]
        opt_settings = opt_dicts[i]["settings_history"][max_id]

        json_data["max_scores"] += [max_score]
        json_data["opt_settings"] += [QNopt.settings_to_list(opt_settings)]

        plt.plot(
            opt_dicts[i]["samples"],
            opt_dicts[i]["scores"],
            "--.",
            label="{:.2f}".format(noise_param),
        )
        plt.plot([max_sample], [max_score], "r*")

    print(json_data["max_scores"])

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    filename = "script/data/chsh-breaking_depolarizing/" + ansatz_name + "/" + datetime_ext
    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_data))

    plt.plot(
        opt_dicts[0]["samples"],
        [2 * np.sqrt(2)] * len(opt_dicts[0]["samples"]),
        label="Quantum Bound",
    )
    plt.plot(opt_dicts[0]["samples"], [2] * len(opt_dicts[0]["samples"]), label="Classical Bound")
    plt.title("One-Sided Depolarizing in CHSH Scenario:\n" + ansatz_name)
    plt.ylabel("CHSH Score")
    plt.xlabel("Epoch")
    plt.legend(ncol=3)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":

    client = Client(processes=True)

    max_entangled_prep = [
        QNopt.PrepareNode(1, [0, 1], QNopt.max_entangled_state, 3),
    ]
    Rot_CNOT_Rot_prep = [
        QNopt.PrepareNode(1, [0, 1], Rot_CNOT_Rot_prep, 12),
    ]
    arb_prep = [QNopt.PrepareNode(1, [0, 1], qml.templates.subroutines.ArbitraryUnitary, 15)]
    meas_nodes = [
        QNopt.MeasureNode(2, 2, [0], Rot_meas_ansatz, 3),
        QNopt.MeasureNode(2, 2, [1], Rot_meas_ansatz, 3),
    ]

    param_range = np.concatenate(
        (np.arange(0, 1.01, 0.05), np.array([(1 - 1 / np.sqrt(2)) * 3 / 4]))
    )

    max_entangled_optimization = chsh_noise_optimization(
        max_entangled_prep, meas_nodes, sample_width=5, step_size=0.2, num_steps=30, verbose=False
    )
    max_entangled_jobs = client.map(max_entangled_optimization, param_range)
    max_entangled_opts = client.gather(max_entangled_jobs)
    save_optimization_results("max_entangled", param_range, max_entangled_opts)

    client.restart()

    rot_cnot_rot_optimization = chsh_noise_optimization(
        Rot_CNOT_Rot_prep, meas_nodes, sample_width=5, step_size=0.2, num_steps=35, verbose=False
    )
    rot_cnot_rot_jobs = client.map(rot_cnot_rot_optimization, param_range)
    rot_cnot_rot_opts = client.gather(rot_cnot_rot_jobs)
    save_optimization_results("rot_cnot_rot", param_range, rot_cnot_rot_opts)

    client.restart()

    arb_optimization = chsh_noise_optimization(
        arb_prep, meas_nodes, sample_width=5, step_size=0.2, num_steps=30, verbose=False
    )
    arb_jobs = client.map(arb_optimization, param_range)
    arb_opts = client.gather(arb_jobs)
    save_optimization_results("arbitrary", param_range, arb_opts)
