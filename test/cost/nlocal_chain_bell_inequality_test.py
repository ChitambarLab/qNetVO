import pytest
from pennylane import numpy as np
import pennylane as qml


import qnetvo as qnet


class TestNLocalChainBellInequality:
    @pytest.mark.parametrize("parallel_flag", [False, True])
    def test_nlocal_chain_cost_22(self, parallel_flag):
        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
        ]

        bilocal_chain_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        bilocal_chain_cost = qnet.nlocal_chain_cost_22(bilocal_chain_ansatz, parallel=parallel_flag)

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

    @pytest.mark.parametrize(
        "prep_nodes,meas_nodes",
        [
            (
                [
                    qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
                    qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
                ],
                [
                    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
                    qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
                    qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
                ],
            ),
            (
                [
                    qnet.PrepareNode(1, [0, 1], qnet.local_RY, 2),
                    qnet.PrepareNode(1, [2, 3], qnet.local_RY, 2),
                    qnet.PrepareNode(1, [4, 5], qnet.local_RY, 2),
                ],
                [
                    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
                    qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
                    qnet.MeasureNode(2, 2, [3, 4], qnet.local_RY, 2),
                    qnet.MeasureNode(2, 2, [5], qnet.local_RY, 1),
                ],
            ),
        ],
    )
    def test_parallel_nlocal_chain_grad_fn(self, prep_nodes, meas_nodes):

        bilocal_chain_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        np.random.seed(45)
        rand_settings = bilocal_chain_ansatz.rand_scenario_settings()

        bilocal_chain_cost = qnet.nlocal_chain_cost_22(bilocal_chain_ansatz)
        grad_match = qml.grad(bilocal_chain_cost, argnum=0)(rand_settings)

        bilocal_chain_grad = qnet.parallel_nlocal_chain_grad_fn(bilocal_chain_ansatz)
        grad = bilocal_chain_grad(rand_settings)

        assert len(grad[0]) == len(grad_match[0])
        for i in range(len(grad[0])):
            assert np.allclose(grad[0][i], grad_match[0][i])

        assert len(grad[1]) == len(grad_match[1])
        for i in range(len(grad[1])):
            assert np.allclose(grad[1][i], grad_match[1][i])

    @pytest.mark.parametrize("parallel_flag", [True, False])
    def test_J22_fn(self, parallel_flag):

        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], qnet.local_RY, 2),
            qnet.PrepareNode(1, [2, 3], qnet.local_RY, 2),
            qnet.PrepareNode(1, [4, 5], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3, 4], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [5], qnet.local_RY, 1),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        J22 = qnet.J22_fn(ansatz, parallel=parallel_flag)

        zero_settings = ansatz.zero_scenario_settings()

        assert np.isclose(0, J22(zero_settings))

        settings = zero_settings
        settings[1][1][0] = [np.pi / 4, np.pi / 3]
        settings[1][2][0] = [np.pi / 5, np.pi / 6]

        assert np.isclose(0, J22(settings))

        settings[1][0][1] = [np.pi]
        settings[1][3][0] = [np.pi]

        assert np.isclose(-4, J22(settings))

        settings[1][1][1] = [np.pi, 0]

        assert np.isclose(4, J22(settings))

    @pytest.mark.parametrize("parallel_flag", [True, False])
    def test_I22_fn(self, parallel_flag):

        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], qnet.local_RY, 2),
            qnet.PrepareNode(1, [2, 3], qnet.local_RY, 2),
            qnet.PrepareNode(1, [4, 5], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3, 4], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [5], qnet.local_RY, 1),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        I22 = qnet.I22_fn(ansatz, parallel=parallel_flag)

        zero_settings = ansatz.zero_scenario_settings()

        assert np.isclose(4, I22(zero_settings))

        settings = zero_settings
        settings[1][1][1] = [np.pi / 4, np.pi / 3]
        settings[1][2][1] = [np.pi / 5, np.pi / 6]

        assert np.isclose(4, I22(settings))

        settings[1][1][0] = [np.pi, 0]

        assert np.isclose(-4, I22(settings))

        settings[1][0][1] = [np.pi]

        assert np.isclose(0, I22(settings))
