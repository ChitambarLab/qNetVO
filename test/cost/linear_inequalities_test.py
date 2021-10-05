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

        settings = network_ansatz.zero_scenario_settings()
        settings[0][0][1, 0] = np.pi
        assert np.isclose(cost(settings), -2)

        settings[0][1][1, :] = [0, np.pi]
        assert np.isclose(cost(settings), -4)

        settings[0][1][2, :] = [np.pi, 0]
        assert np.isclose(cost(settings), -6)

        settings[0][1][3, :] = [np.pi, np.pi]
        assert np.isclose(cost(settings), -8)

        dichotomic_game = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
        dichotomic_cost = QNopt.linear_probs_cost(network_ansatz, dichotomic_game)

        assert np.isclose(dichotomic_cost(zero_settings), -1)
        assert np.isclose(dichotomic_cost(settings), -2)

        # value erros
        with pytest.raises(
            ValueError,
            match="``linear_probs_cost`` does not currently support coarse-graining from 8 -> 3 outputs.",
        ):
            game = np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]]
            )
            cost = QNopt.linear_probs_cost(network_ansatz, game)
            cost(zero_settings)
