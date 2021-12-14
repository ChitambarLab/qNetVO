import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestLinearInequalityCost:
    def example_ansatz(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(4, [1, 2], QNopt.local_RY, 2),
        ]
        meas_nodes = [
            QNopt.MeasureNode(1, 2, [0, 1, 2], QNopt.local_RY, 3),
        ]

        return QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

    def test_linear_probs_inequality_cost_no_post_processing(self):
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

        cost = QNopt.linear_probs_cost_fn(network_ansatz, game)

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

    def test_linear_probs_cost_dichotomic_game(self):
        network_ansatz = self.example_ansatz()

        zero_settings = network_ansatz.zero_scenario_settings()
        settings = network_ansatz.zero_scenario_settings()
        settings[0][0][1, 0] = np.pi
        settings[0][1][1, :] = [0, np.pi]

        dichotomic_game = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
        dichotomic_cost = QNopt.linear_probs_cost_fn(
            network_ansatz,
            dichotomic_game,
            post_map=np.array([[1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]]),
        )

        assert np.isclose(dichotomic_cost(zero_settings), -1)
        assert np.isclose(dichotomic_cost(settings), -2)

    @pytest.mark.parametrize(
        "game,post_map,match",
        [
            ([], [], r"The `game` matrix must either have 8 rows, or a `post_map` is needed\."),
            (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
                [],
                r"The `game` matrix must have 8 columns\.",
            ),
            (
                [],
                np.array([[1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 1]]),
                r"The `post_map` must have 3 rows\.",
            ),
            (
                [],
                np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]),
                r"The `post_map` must have 8 columns\.",
            ),
        ],
    )
    def test_linear_probs_cost_fn_errors(self, game, post_map, match):
        ansatz = self.example_ansatz()

        default_game = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]]
        )

        game = default_game if len(game) == 0 else game
        with pytest.raises(ValueError, match=match):
            QNopt.linear_probs_cost_fn(ansatz, game, post_map=post_map)
