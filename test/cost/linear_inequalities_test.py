import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestLinearInequalityCost:
    def test_linear_probs_inequality_cost(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(4, [1, 2], QNopt.local_RY, 2),
        ]

        meas_nodes = [
            QNopt.MeasureNode(1, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(1, 4, [1, 2], QNopt.local_RY, 2),
        ]

        network_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        game = np.eye(8)

        cost = QNopt.linear_probs_cost(network_ansatz, game)

        zero_settings = network_ansatz.zero_scenario_settings()
        assert np.isclose(cost(zero_settings), -1)

        settings = zero_settings
        settings[0][0][1, 0] = np.pi
        assert np.isclose(cost(settings), -2)

        settings[0][1][1, :] = [0, np.pi]
        assert np.isclose(cost(settings), -4)

        settings[0][1][2, :] = [np.pi, 0]
        assert np.isclose(cost(settings), -6)

        settings[0][1][3, :] = [np.pi, np.pi]
        assert np.isclose(cost(settings), -8)

    def test_mixed_base_convert(self):

        assert np.all(QNopt.mixed_base_convert(0, [2, 2]) == [0, 0])
        assert np.all(QNopt.mixed_base_convert(2, [2, 2]) == [1, 0])
        assert np.all(QNopt.mixed_base_convert(9, [2, 3, 4]) == [0, 2, 1])
        assert np.all(QNopt.mixed_base_convert(119, [5, 4, 3, 2, 1]) == [4, 3, 2, 1, 0])
