import pytest
from pennylane import numpy as np

from context import qnetvo as qnet


class TestCHSHInequalityCost:
    @pytest.mark.parametrize("parallel_flag", [False, True])
    def test_chsh_inequality_cost(self, parallel_flag):

        prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
        ]

        chsh_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        chsh_cost = qnet.chsh_inequality_cost(chsh_ansatz, parallel=parallel_flag)

        zero_settings = chsh_ansatz.zero_scenario_settings()
        assert np.isclose(chsh_cost(zero_settings), -2)

        settings = zero_settings
        settings[1][0][1] = np.pi / 2
        settings[1][1][0] = np.pi / 4
        settings[1][1][1] = -np.pi / 4

        assert np.isclose(chsh_cost(settings), -2 * np.sqrt(2))
