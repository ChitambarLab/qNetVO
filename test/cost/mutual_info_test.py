import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


class TestMutualInfoCostFn:
    @pytest.mark.parametrize(
        "network_settings,priors,postmap,match",
        [
            (
                [0, 0, 0, 0, 0, 0],
                [np.ones(3) / 3],
                np.array([[1, 0], [0, 1], [0, 1]]),
                0.0,
            ),
            (
                [0, np.pi, 0, 0, -np.pi, 0],
                [np.ones(3) / 3],
                np.array([[1, 0], [0, 1], [0, 1]]),
                -0.9182958,
            ),
            (
                [0, np.pi, 0, 0, -np.pi, 0],
                [np.array([0.5, 0.5, 0])],
                np.array([[1, 0], [0, 1], [0, 1]]),
                -1,
            ),
        ],
    )
    def test_mutual_info_cost_qubit_33(self, network_settings, priors, postmap, match):
        ansatz = qnet.NetworkAnsatz(
            [qnet.PrepareNode(3, [0], qnet.local_RY, 1)],
            [qnet.MeasureNode(1, 3, [0], qnet.local_RY, 1)],
        )

        mutual_info = qnet.mutual_info_cost_fn(ansatz, priors, postmap=postmap)

        assert np.isclose(mutual_info(*network_settings), match)

    def test_mutual_info_2_senders(self):
        ansatz = qnet.NetworkAnsatz(
            [
                qnet.PrepareNode(2, [0], qnet.local_RY, 1),
                qnet.PrepareNode(2, [1], qnet.local_RY, 1),
            ],
            [qnet.MeasureNode(1, 4, [0, 1], qnet.local_RY, 2)],
        )

        network_settings = [0, np.pi, 0, np.pi, 0, 0]
        priors = [np.ones(2) / 2, np.ones(2) / 2]
        mutual_info = qnet.mutual_info_cost_fn(ansatz, priors)

        assert np.isclose(mutual_info(*network_settings), -2)


class TestMutualInfoOptimimzation:
    @pytest.mark.parametrize(
        "priors,match", [([np.ones(3) / 3], 0.9182), ([np.array([0.5, 0.5, 0])], 1)]
    )
    def test_mutual_info_opt_qubit_33(self, priors, match):
        ansatz = qnet.NetworkAnsatz(
            [qnet.PrepareNode(3, [0], qnet.local_RY, 1)],
            [qnet.MeasureNode(1, 3, [0], qnet.local_RY, 1)],
        )

        postmap = np.array([[1, 0], [0, 1], [0, 1]])

        mutual_info = qnet.mutual_info_cost_fn(ansatz, priors, postmap=postmap)

        np.random.seed(12)
        network_settings = ansatz.rand_network_settings()

        opt_dict = qnet.gradient_descent(
            mutual_info, network_settings, step_size=0.1, num_steps=28, sample_width=24
        )

        assert np.isclose(opt_dict["scores"][-1], match, atol=0.0005)


class TestShannonEntropy:
    def test_shannon_entropy_pure_state(self):
        np.random.seed(123)

        def test_meas_circ(settings, wires):
            qml.CNOT(wires=wires[0:2])
            qml.RZ(settings[0], wires=wires[0])
            qml.RY(settings[1], wires=wires[0])

        prep_node = [qnet.PrepareNode(wires=[0, 1], ansatz_fn=qnet.ghz_state)]
        meas_node = [
            qnet.MeasureNode(num_out=4, wires=[0, 1], ansatz_fn=test_meas_circ, num_settings=2)
        ]

        ansatz = qnet.NetworkAnsatz(prep_node, meas_node)
        shannon_entropy = qnet.shannon_entropy_cost_fn(ansatz)

        settings = ansatz.rand_network_settings()
        opt_dict = qnet.gradient_descent(
            shannon_entropy,
            settings,
            step_size=0.15,
            sample_width=5,
            num_steps=35,
        )

        assert np.isclose(opt_dict["scores"][-1], 0, atol=0.0005)

    def test_von_neumann_entropy_mixed_state(self):
        np.random.seed(123)

        prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
        meas_nodes = [qnet.MeasureNode(1, 4, [0, 1], qml.ArbitraryUnitary, 4**2 - 1)]
        gamma = 0.04
        noise_nodes = [
            qnet.NoiseNode([0], lambda settings, wires: qml.DepolarizingChannel(gamma, wires)),
            qnet.NoiseNode([1], lambda settings, wires: qml.DepolarizingChannel(gamma, wires)),
        ]

        ansatz = qnet.NetworkAnsatz(
            prep_nodes, noise_nodes, meas_nodes, dev_kwargs={"name": "default.mixed"}
        )
        shannon_entropy = qnet.shannon_entropy_cost_fn(ansatz)

        settings = ansatz.rand_network_settings()
        opt_dict = qnet.gradient_descent(
            shannon_entropy, settings, step_size=0.1, sample_width=5, num_steps=30
        )

        assert np.isclose(opt_dict["scores"][-1], -0.518, atol=0.0005)
