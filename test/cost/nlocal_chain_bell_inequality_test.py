import pytest
from pennylane import numpy as np
import pennylane as qml

import qnetvo as qnet


class TestNLocalChainBellInequality:
    @pytest.mark.parametrize("parallel_flag", [False, True])
    def test_nlocal_chain_22_cost_fn(self, parallel_flag):
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

        bilocal_chain_cost = qnet.nlocal_chain_22_cost_fn(
            bilocal_chain_ansatz, parallel=parallel_flag
        )

        zero_settings = bilocal_chain_ansatz.zero_network_settings()

        assert np.isclose(bilocal_chain_cost(*zero_settings), -1)

        ideal_settings = [0, np.pi / 2, np.pi / 4, 0, -np.pi / 4, np.pi / 2, np.pi / 4, -np.pi / 4]

        assert np.isclose(bilocal_chain_cost(*ideal_settings), -(1 / np.sqrt(2) + 1 / np.sqrt(2)))

    @pytest.mark.parametrize("natural_grad", [True, False])
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
    def test_parallel_nlocal_chain_grad_fn(self, natural_grad, prep_nodes, meas_nodes):

        chain_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        np.random.seed(45)
        rand_settings = chain_ansatz.rand_network_settings()

        chain_cost = qnet.nlocal_chain_22_cost_fn(chain_ansatz)
        grad_match = qml.grad(chain_cost)(*rand_settings)

        chain_grad = qnet.parallel_nlocal_chain_grad_fn(chain_ansatz, natural_grad=natural_grad)
        grad = chain_grad(*rand_settings)

        assert len(grad_match) == len(grad)
        if not (natural_grad):
            assert np.allclose(grad, grad_match)

    @pytest.mark.parametrize("parallel_flag", [True, False])
    def test_chain_J22_fn(self, parallel_flag):

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

        J22 = qnet.chain_J22_fn(ansatz, parallel=parallel_flag)

        zero_settings = ansatz.zero_network_settings()
        assert np.isclose(0, J22(*zero_settings))

        settings = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            np.pi / 4,
            np.pi / 3,
            0,
            0,
            np.pi / 5,
            np.pi / 6,
            0,
            0,
            0,
            0,
        ]
        assert np.isclose(0, J22(*settings))

        settings[7] = np.pi
        settings[16] = np.pi
        assert np.isclose(-4, J22(*settings))

        settings[10:12] = [np.pi, 0]
        assert np.isclose(4, J22(*settings))

    @pytest.mark.parametrize("parallel_flag", [True, False])
    def test_chain_I22_fn(self, parallel_flag):

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

        I22 = qnet.chain_I22_fn(ansatz, parallel=parallel_flag)

        zero_settings = ansatz.zero_network_settings()
        assert np.isclose(4, I22(*zero_settings))

        settings = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            np.pi / 4,
            np.pi / 3,
            0,
            0,
            np.pi / 5,
            np.pi / 6,
            0,
            0,
        ]
        assert np.isclose(4, I22(*settings))

        settings[8:10] = [np.pi, 0]
        assert np.isclose(-4, I22(*settings))

        settings[7] = np.pi
        assert np.isclose(0, I22(*settings))
