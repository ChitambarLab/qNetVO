import pytest
import pennylane as qml
from pennylane import numpy as np

from context import qnetvo as QNopt


class TestCostMagicSquaresGame:
    def test_magic_squares_game_cost(self):

        prep_nodes = [QNopt.PrepareNode(1, range(4), QNopt.bell_state_copies, 0)]
        meas_nodes = [
            QNopt.MeasureNode(3, 4, [0, 1], qml.templates.subroutines.ArbitraryUnitary, 15),
            QNopt.MeasureNode(3, 4, [2, 3], qml.templates.subroutines.ArbitraryUnitary, 15),
        ]
        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        np.random.seed(1)
        settings = ansatz.rand_scenario_settings()
        cost = QNopt.magic_squares_game_cost(ansatz)

        opt_dict = QNopt.gradient_descent(cost, settings, step_size=4.5, num_steps=15)
        assert np.isclose(opt_dict["opt_score"], 1, atol=0.01)
