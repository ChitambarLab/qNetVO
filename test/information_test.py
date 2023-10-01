import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


class TestBehaviorFn:
    def test_simple_settings(self):
        prep_nodes = [
            qnet.PrepareNode(num_in=2, wires=[0], ansatz_fn=qnet.local_RY, num_settings=1),
            qnet.PrepareNode(num_in=2, wires=[1], ansatz_fn=qnet.local_RY, num_settings=1),
        ]
        meas_nodes = [
            qnet.MeasureNode(
                num_in=2, num_out=2, wires=[0], ansatz_fn=qnet.local_RY, num_settings=1
            ),
            qnet.MeasureNode(
                num_in=2, num_out=2, wires=[1], ansatz_fn=qnet.local_RY, num_settings=1
            ),
        ]
        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        P_Net = qnet.behavior_fn(ansatz)
        zero_settings = ansatz.zero_network_settings()

        assert np.all(
            P_Net(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        settings = zero_settings
        settings[1] = np.pi

        assert np.allclose(
            P_Net(settings),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

        settings[7] = np.pi / 2

        assert np.allclose(
            P_Net(settings),
            [
                [1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5],
            ],
        )

    def test_42_coarse_grain(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(2, [1], qnet.local_RY, 1),
        ]
        meas_nodes = [
            qnet.MeasureNode(4, 2, [0, 1], qnet.local_RY, 2),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        P_Net_no_postmap = qnet.behavior_fn(ansatz)
        zero_settings = ansatz.zero_network_settings()
        assert np.all(
            P_Net_no_postmap(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        with pytest.raises(ValueError, match=r"The `postmap` must have 4 columns\."):
            qnet.behavior_fn(ansatz, postmap=np.array([[1, 0], [0, 1]]))

        P_Net = qnet.behavior_fn(ansatz, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]))
        assert np.all(
            P_Net(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        settings = zero_settings
        settings[1] = np.pi

        assert np.allclose(
            P_Net(settings),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        )

    def test_rand_settings(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(3, [1], qnet.local_RY, 1),
            qnet.PrepareNode(4, [2, 3], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
            qnet.MeasureNode(3, 4, [2, 3], qnet.local_RY, 2),
        ]
        net_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        net_behavior = qnet.behavior_fn(net_ansatz)

        np.random.seed(419)
        rand_settings = net_ansatz.rand_network_settings()

        P_Net = net_behavior(rand_settings)

        assert P_Net.shape == (16, 288)
        assert np.allclose(np.ones(288), [np.sum(P_Net[:, i]) for i in range(288)])

    def test_inputs_from_multiple_layers(self):
        prep_nodes_a = [
            qnet.PrepareNode(num_in=2, wires=[0], ansatz_fn=qnet.local_RY, num_settings=1),
        ]
        prep_nodes_b = [
            qnet.PrepareNode(num_in=2, wires=[1], ansatz_fn=qnet.local_RY, num_settings=1),
        ]

        meas_nodes = [
            qnet.MeasureNode(
                num_in=2, num_out=2, wires=[0], ansatz_fn=qnet.local_RY, num_settings=1
            ),
            qnet.MeasureNode(
                num_in=2, num_out=2, wires=[1], ansatz_fn=qnet.local_RY, num_settings=1
            ),
        ]
        ansatz = qnet.NetworkAnsatz(prep_nodes_a, prep_nodes_b, meas_nodes)
        P_Net = qnet.behavior_fn(ansatz)
        zero_settings = ansatz.zero_network_settings()

        assert np.all(
            P_Net(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        settings = zero_settings
        settings[1] = np.pi

        assert np.allclose(
            P_Net(settings),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

        settings[7] = np.pi / 2

        assert np.allclose(
            P_Net(settings),
            [
                [1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5],
            ],
        )


class TestShannonEntropy:
    @pytest.mark.parametrize(
        "probs, entropy_match",
        [([0, 0, 0, 1], 0), (np.ones(4) / 4, 2), ([0, -0.00000000001, 0, 1], 0)],
    )
    def test_simple_shannon_entropy_cases(self, probs, entropy_match):
        assert qnet.shannon_entropy(probs) == entropy_match
