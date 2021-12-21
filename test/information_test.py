import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestBehaviorFn:
    def test_simple_settings(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(2, [1], QNopt.local_RY, 1),
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
        ]
        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        P_Net = QNopt.behavior_fn(ansatz)
        zero_settings = ansatz.zero_scenario_settings()

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
        settings[0][0][1] = [np.pi]

        assert np.allclose(
            P_Net(settings),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

        settings[1][1][1] = [np.pi / 2]

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
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(2, [1], QNopt.local_RY, 1),
        ]
        meas_nodes = [
            QNopt.MeasureNode(4, 2, [0, 1], QNopt.local_RY, 2),
        ]
        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        P_Net_no_post_map = QNopt.behavior_fn(ansatz)
        zero_settings = ansatz.zero_scenario_settings()
        assert np.all(
            P_Net_no_post_map(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        with pytest.raises(ValueError, match=r"The `post_map` must have 4 columns\."):
            QNopt.behavior_fn(ansatz, post_map=np.array([[1, 0], [0, 1]]))

        P_Net = QNopt.behavior_fn(ansatz, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]))
        assert np.all(
            P_Net(zero_settings)
            == [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        settings = zero_settings
        settings[0][0][1] = [np.pi]

        assert np.allclose(
            P_Net(settings),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        )

    def test_rand_settings(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(3, [1], QNopt.local_RY, 1),
            QNopt.PrepareNode(4, [2, 3], QNopt.local_RY, 2),
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
            QNopt.MeasureNode(3, 4, [2, 3], QNopt.local_RY, 2),
        ]
        net_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        net_behavior = QNopt.behavior_fn(net_ansatz)

        np.random.seed(419)
        rand_settings = net_ansatz.rand_scenario_settings()

        P_Net = net_behavior(rand_settings)

        assert P_Net.shape == (16, 288)
        assert np.allclose(np.ones(288), [np.sum(P_Net[:, i]) for i in range(288)])


class TestShannonEntropy:
    @pytest.mark.parametrize(
        "probs, entropy_match",
        [([0, 0, 0, 1], 0), (np.ones(4) / 4, 2), ([0, -0.00000000001, 0, 1], 0)],
    )
    def test_simple_shannon_entropy_cases(self, probs, entropy_match):
        assert QNopt.shannon_entropy(probs) == entropy_match
