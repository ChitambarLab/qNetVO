import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


class TestCostI3322BellInequality:
    def test_post_process_I_3322_joint_probs(self):
        probs_vec = np.array([0.1, 0.2, 0.3, 0.4])
        P00xy = qnet.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 0.1

        probs_vec = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]) / 4
        P00xy = qnet.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 1

        probs_vec = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]) / 12
        P00xy = qnet.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 0

    def test_I_3322_bell_inequality_cost_fn_qubit_optimization(self):
        def local_RY(settings, wires):
            for i, wire in enumerate(wires):
                qml.RY(settings[i], wires=wire)

        prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
        meas_nodes = [
            qnet.MeasureNode(3, 2, [0], local_RY, 1),
            qnet.MeasureNode(3, 2, [1], local_RY, 1),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        I_3322_cost = qnet.I_3322_bell_inequality_cost_fn(ansatz)

        np.random.seed(13)
        init_settings = ansatz.rand_network_settings()

        opt_dict = qnet.gradient_descent(
            I_3322_cost, init_settings, step_size=0.5, num_steps=50, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], 0.25, atol=10e-6)

        # example close to optimal strategy
        settings = [
            3 * np.pi / 4,
            -14 * np.pi / 15,
            2 * np.pi / 5,
            3 * np.pi / 4,
            16 * np.pi / 15,
            2 * np.pi / 5,
        ]

        assert np.isclose(I_3322_cost(*settings), -0.25, atol=1e-3)
