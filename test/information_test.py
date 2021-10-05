import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestNetworkBehaviorFn:
    def test_simple_settings(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(2, [1], QNopt.local_RY, 1),
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
        ]
        net_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        P_Net = QNopt.network_behavior_fn(net_ansatz)
        zero_settings = net_ansatz.zero_scenario_settings()

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

    def test_rand_settings(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(3, [1], QNopt.local_RY, 1),
            QNopt.PrepareNode(4, [2,3], QNopt.local_RY, 2)
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
            QNopt.MeasureNode(3, 4, [2,3], QNopt.local_RY, 2)
        ]
        net_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        net_behavior = QNopt.network_behavior_fn(net_ansatz)

        np.random.seed(419)
        rand_settings = net_ansatz.rand_scenario_settings()

        P_Net = net_behavior(rand_settings)

        assert P_Net.shape == (16,288)
        assert np.allclose(np.ones(288), [np.sum(P_Net[:,i]) for i in range(288)])


class TestBisenderMACMutualInfo:

    @pytest.mark.parametrize(
        "mac_behavior, priors_x, priors_y, exp_rates_tuple",
        [
            (
                np.array([
                    [1,1,1,1],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]
                ]), np.ones(2)/2, np.ones(2)/2, (0,0,0)
            ),
            (
                np.array([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]
                ]), np.ones(2)/2, np.ones(2)/2, (1,1,2)
            ),
            (
                np.array([
                    [1,0,0,1],
                    [0,1,1,0],
                ]), np.ones(2)/2, np.ones(2)/2, (1,1,1)
            ),
        ]
    )
    def test_bisender_mac_mutual_info(self, mac_behavior, priors_x, priors_y, exp_rates_tuple):
        rates_tuple = QNopt.bisender_mac_mutual_info(mac_behavior, priors_x, priors_y)
        assert np.allclose(rates_tuple, exp_rates_tuple)


class TestShannonEntropy:
    @pytest.mark.parametrize(
        "probs, entropy_match",
        [
            ([0,0,0,1], 0),
            (np.ones(4)/4, 2)
        ]
    )
    def test_simple_shannon_entropy_cases(self, probs, entropy_match):
        assert QNopt.shannon_entropy(probs) == entropy_match
