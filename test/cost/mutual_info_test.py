import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


class TestMutualInfoCostFn:
    @pytest.mark.parametrize(
        "scenario_settings,priors,postmap,match",
        [
            (
                [[np.zeros((3, 1))], [np.zeros((3, 1))]],
                [np.ones(3) / 3],
                np.array([[1, 0], [0, 1], [0, 1]]),
                0.0,
            ),
            (
                [[np.array([[0], [np.pi], [0]])], [np.array([[0], [-np.pi], [0]])]],
                [np.ones(3) / 3],
                np.array([[1, 0], [0, 1], [0, 1]]),
                -0.9182958,
            ),
            (
                [[np.array([[0], [np.pi], [0]])], [np.array([[0], [-np.pi], [0]])]],
                [np.array([0.5, 0.5, 0])],
                np.array([[1, 0], [0, 1], [0, 1]]),
                -1,
            ),
        ],
    )
    def test_mutual_info_cost_qubit_33(self, scenario_settings, priors, postmap, match):

        ansatz = qnet.NetworkAnsatz(
            [qnet.PrepareNode(3, [0], qnet.local_RY, 1)],
            [qnet.MeasureNode(1, 3, [0], qnet.local_RY, 1)],
        )

        mutual_info = qnet.mutual_info_cost_fn(ansatz, priors, postmap=postmap)

        assert np.isclose(mutual_info(scenario_settings), match)

    def test_mutual_info_2_senders(self):

        ansatz = qnet.NetworkAnsatz(
            [
                qnet.PrepareNode(2, [0], qnet.local_RY, 1),
                qnet.PrepareNode(2, [1], qnet.local_RY, 1),
            ],
            [qnet.MeasureNode(1, 4, [0, 1], qnet.local_RY, 2)],
        )

        scenario_settings = [
            [np.array([[0], [np.pi]]), np.array([[0], [np.pi]])],
            [np.array([[0, 0]])],
        ]

        priors = [np.ones(2) / 2, np.ones(2) / 2]

        mutual_info = qnet.mutual_info_cost_fn(ansatz, priors)

        assert np.isclose(mutual_info(scenario_settings), -2)


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
        scenario_settings = ansatz.rand_scenario_settings()

        opt_dict = qnet.gradient_descent(
            mutual_info, scenario_settings, step_size=0.1, num_steps=28, sample_width=24
        )

        assert np.isclose(opt_dict["scores"][-1], match, atol=0.0005)


class TestShannonEntropy:
    def test_shannon_entropy_pure_state(self):
        np.random.seed(123)

        prep_node = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
        meas_node = [qnet.MeasureNode(1, 4, [0, 1], qml.ArbitraryUnitary, 4**2 - 1)]

        ansatz = qnet.NetworkAnsatz(prep_node, meas_node)
        shannon_entropy = qnet.shannon_entropy_cost_fn(ansatz)

        settings = ansatz.rand_scenario_settings()
        opt_dict = qnet.gradient_descent(
            shannon_entropy, settings, step_size=0.08, sample_width=5, num_steps=30
        )

        assert np.isclose(opt_dict["scores"][-1], 0, atol=0.0005)

    def test_von_neumann_entropy_mixed_state(self):
        np.random.seed(123)

        prep_node = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
        meas_node = [qnet.MeasureNode(1, 4, [0, 1], qml.ArbitraryUnitary, 4**2 - 1)]
        gamma = 0.04
        noise_node = [
            qnet.NoiseNode([0], lambda settings, wires: qml.DepolarizingChannel(gamma, wires)),
            qnet.NoiseNode([1], lambda settings, wires: qml.DepolarizingChannel(gamma, wires)),
        ]

        ansatz = qnet.NetworkAnsatz(prep_node, meas_node, noise_node)
        shannon_entropy = qnet.shannon_entropy_cost_fn(ansatz)

        settings = ansatz.rand_scenario_settings()
        opt_dict = qnet.gradient_descent(
            shannon_entropy, settings, step_size=0.1, sample_width=5, num_steps=30
        )

        assert np.isclose(opt_dict["scores"][-1], -0.518, atol=0.0005)
