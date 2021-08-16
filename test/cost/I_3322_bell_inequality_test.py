import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestCostI3322BellInequality:
    def test_post_process_I_3322_joint_probs(self):

        probs_vec = np.array([0.1, 0.2, 0.3, 0.4])
        P00xy = QNopt.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 0.1

        probs_vec = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]) / 4
        P00xy = QNopt.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 1

        probs_vec = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]) / 12
        P00xy = QNopt.post_process_I_3322_joint_probs(probs_vec)
        assert P00xy == 0

    def test_I_3322_bell_inequality_cost_qubit_optimization(self):
        def local_RY(settings, wires):
            for i, wire in enumerate(wires):
                qml.RY(settings[i], wires=wire)

        prep_nodes = [QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0)]
        meas_nodes = [
            QNopt.MeasureNode(3, 2, [0], local_RY, 1),
            QNopt.MeasureNode(3, 2, [1], local_RY, 1),
        ]

        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        I_3322_cost = QNopt.I_3322_bell_inequality_cost(ansatz)

        np.random.seed(13)
        init_settings = ansatz.rand_scenario_settings()

        opt_dict = QNopt.gradient_descent(
            I_3322_cost, init_settings, step_size=0.5, num_steps=50, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], 0.25, atol=10e-6)

        # example close to optimal strategy
        settings = ansatz.zero_scenario_settings()
        settings[1][0][0] = [3 * np.pi / 4]
        settings[1][0][1] = [-14 * np.pi / 15]
        settings[1][0][2] = [2 * np.pi / 5]
        settings[1][1][0] = [3 * np.pi / 4]
        settings[1][1][1] = [16 * np.pi / 15]
        settings[1][1][2] = [2 * np.pi / 5]

        assert np.isclose(I_3322_cost(settings), -0.25, atol=1e-3)
