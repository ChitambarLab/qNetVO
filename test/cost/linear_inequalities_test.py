import pytest
import pennylane as qml
from pennylane import numpy as np

from context import qnetvo as qnet


class TestLinearInequalityCost:
    def example_ansatz(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(4, [1, 2], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(1, 2, [0, 1, 2], qnet.local_RY, 3),
        ]

        return qnet.NetworkAnsatz(prep_nodes, meas_nodes)

    def test_linear_probs_inequality_cost_no_post_processing(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(4, [1, 2], qnet.local_RY, 2),
        ]

        meas_nodes = [
            qnet.MeasureNode(1, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(1, 4, [1, 2], qnet.local_RY, 2),
        ]

        network_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        game = np.eye(8)

        cost = qnet.linear_probs_cost_fn(network_ansatz, game)

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
        dichotomic_cost = qnet.linear_probs_cost_fn(
            network_ansatz,
            dichotomic_game,
            postmap=np.array([[1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]]),
        )

        assert np.isclose(dichotomic_cost(zero_settings), -1)
        assert np.isclose(dichotomic_cost(settings), -2)

    @pytest.mark.parametrize(
        "game,postmap,match",
        [
            ([], [], r"The `game` matrix must either have 8 rows, or a `postmap` is needed\."),
            (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
                [],
                r"The `game` matrix must have 8 columns\.",
            ),
            (
                [],
                np.array([[1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 1]]),
                r"The `postmap` must have 3 rows\.",
            ),
            (
                [],
                np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]),
                r"The `postmap` must have 8 columns\.",
            ),
        ],
    )
    def test_linear_probs_cost_fn_errors(self, game, postmap, match):
        ansatz = self.example_ansatz()

        default_game = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]]
        )

        game = default_game if len(game) == 0 else game
        with pytest.raises(ValueError, match=match):
            qnet.linear_probs_cost_fn(ansatz, game, postmap=postmap)
