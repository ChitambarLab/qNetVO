import pytest
from pennylane import numpy as np


from context import QNetOptimizer as QNopt


class TestNLocalChainBellInequality:
    @pytest.mark.parametrize("parallel_flag", [False, True])
    def test_nlocal_chain_cost_22(self, parallel_flag):
        prep_nodes = [
            QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0),
            QNopt.PrepareNode(1, [2, 3], QNopt.ghz_state, 0),
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1, 2], QNopt.local_RY, 2),
            QNopt.MeasureNode(2, 2, [3], QNopt.local_RY, 1),
        ]

        bilocal_chain_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        bilocal_chain_cost = QNopt.nlocal_chain_cost_22(bilocal_chain_ansatz, parallel=parallel_flag)

        zero_settings = bilocal_chain_ansatz.zero_scenario_settings()

        assert np.isclose(bilocal_chain_cost(zero_settings), -1)

        ideal_settings = bilocal_chain_ansatz.zero_scenario_settings()
        ideal_settings[1][0][1, 0] = np.pi / 2

        ideal_settings[1][1][0, 0] = np.pi / 4
        ideal_settings[1][1][1, 0] = -np.pi / 4

        ideal_settings[1][1][1, 1] = np.pi / 2

        ideal_settings[1][2][0, 0] = np.pi / 4
        ideal_settings[1][2][1, 0] = -np.pi / 4

        assert np.isclose(bilocal_chain_cost(ideal_settings), -(1 / np.sqrt(2) + 1 / np.sqrt(2)))
