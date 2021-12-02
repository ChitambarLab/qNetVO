from dask.distributed import Client
import time
from context import QNetOptimizer as QNopt
from pennylane import numpy as np
import pennylane as qml

"""
This script attempts to reproduce a result from the literature where two quantum
with broken nonlocality can be used to violate the CHSH inequality.
Unfortunately, we are unable to find a positive case demonstrating the result.
"""

def chsh_noise_optimization(noise_args):
    def prep_ansatz(settings, wires):
        QNopt.max_entangled_state(settings[0:3], wires=[wires[0], wires[2]])
        QNopt.max_entangled_state(settings[3:6], wires=[wires[1], wires[3]])

    def state_ansatz(settings, wires):
        # qml.templates.subroutines.ArbitraryUnitary(settings[0:15],wires=wires[0:2])
        qml.RY(settings[0], wires=wires[0])
        # qml.RZ(settings[1], wires=wires[0])
        qml.RY(settings[1], wires=wires[1])
        # qml.RZ(settings[3], wires=wires[1])
        qml.CNOT(wires=wires[0:2])
        qml.Rot(*settings[2:5], wires=wires[0])
        # qml.RY(settings[4], wires=wires[0])
        # qml.RZ(settings[5], wires=wires[0])
        # qml.RY(settings[6], wires=wires[1])
        # qml.RZ(settings[7], wires=wires[1])
        # qml.CNOT(wires=wires[0:2])

        # qml.Rot(*settings[0:3],wires=wires[0])
        # qml.Rot(*settings[3:6],wires=wires[1])
        # qml.CNOT(wires=wires[0:2])
        # qml.Rot(*settings[6:9],wires=wires[0])
        # qml.Rot(*settings[9:12],wires=wires[1])
        # qml.CNOT(wires=[wires[1],wires[0]])

    def prep_ansatz(settings, wires):
        qml.RY(settings[0], wires=wires[0])
        qml.CNOT(wires=wires[0:2])
        qml.Rot(*settings[1:4], wires=wires[0])
        qml.Rot(*settings[4:7], wires=wires[1])

    def state_ansatz2(settings, wires):
        # state_ansatz(settings[0:5],wires[0:2])
        # state_ansatz(settings[5:10],wires[2:4])
        qml.templates.subroutines.ArbitraryUnitary(settings[0:15], wires=wires[0:2])
        qml.templates.subroutines.ArbitraryUnitary(settings[15:30], wires=wires[2:4])
        # prep_ansatz(settings[0:7], wires=wires[0:2])
        # prep_ansatz(settings[7:14], wires=wires[2:4])

    def local_rot(settings, wires):
        qml.Rot(*settings[0:3], wires=wires[0])
        qml.Rot(*settings[3:6], wires=wires[1])

    def meas_ansatzA(settings, wires):
        qml.Rot(*settings[0:3], wires=wires[0])
        qml.CNOT(wires=wires[0:2])
        qml.RY(settings[3], wires=wires[0])
        qml.RY(settings[4], wires=wires[1])

    def meas_ansatzB(settings, wires):
        qml.Rot(*settings[0:3], wires=wires[1])
        qml.CNOT(wires=wires[0:2])
        qml.RY(settings[3], wires=wires[0])
        qml.RY(settings[4], wires=wires[1])

    prep_nodes = [
        QNopt.PrepareNode(1, [0, 1, 2, 3], state_ansatz2, 30),
    ]
    meas_nodes = [
        QNopt.MeasureNode(2, 2, [0, 2], qml.templates.subroutines.ArbitraryUnitary, 15),
        QNopt.MeasureNode(2, 2, [1, 3], qml.templates.subroutines.ArbitraryUnitary, 15),
    ]

    noise_nodes = [
        QNopt.NoiseNode(
            [0], lambda settings, wires: qml.AmplitudeDamping(noise_args[0], wires=wires[0])
        ),
        QNopt.NoiseNode(
            [3], lambda settings, wires: qml.AmplitudeDamping(noise_args[1], wires=wires[0])
        ),
    ]

    # analytic device approach
    chsh_ansatz1 = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, noise_nodes)
   
   	# stochastic gradient approach
    chsh_ansatz2 = QNopt.NetworkAnsatz(
        prep_nodes,
        meas_nodes,
        noise_nodes,
        dev_kwargs={
            "name": "default.mixed",
            "shots": 700,
        },
    )

    chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz1)
    grad = QNopt.parallel_chsh_grad(chsh_ansatz2, diff_method="parameter-shift")
    # grad = QNopt.chsh_natural_grad(chsh_ansatz2, diff_method="parameter-shift")

    opt_dict = QNopt.gradient_descent(
        chsh_cost,
        chsh_ansatz1.rand_scenario_settings(),
        sample_width=5,
        step_size=0.01,
        num_steps=300,
        grad_fn=grad,
    )

    print("noise aregs : ", noise_args)
    print("max score : ", opt_dict["opt_score"])

    return opt_dict


if __name__ == "__main__":

    client = Client(processes=True)

    A = client.map(
        chsh_noise_optimization, [[0.50001, 0.50001], [0.500001, 0.5], [0.5, 0.50001], [0.5, 0.5]]
    )

    a = time.time()
    opts = client.gather(A)

    print(opts[3]["opt_score"])
    print(opts[3]["opt_settings"])
    print(time.time() - a)
