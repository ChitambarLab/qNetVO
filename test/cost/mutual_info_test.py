import pytest
import pennylane as qml
from pennylane import numpy as np

from context import qnetvo as QNopt


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

        ansatz = QNopt.NetworkAnsatz(
            [QNopt.PrepareNode(3, [0], QNopt.local_RY, 1)],
            [QNopt.MeasureNode(1, 3, [0], QNopt.local_RY, 1)],
        )

        mutual_info = QNopt.mutual_info_cost_fn(ansatz, priors, postmap=postmap)

        assert np.isclose(mutual_info(scenario_settings), match)

    def test_mutual_info_2_senders(self):

        ansatz = QNopt.NetworkAnsatz(
            [
                QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
                QNopt.PrepareNode(2, [1], QNopt.local_RY, 1),
            ],
            [QNopt.MeasureNode(1, 4, [0, 1], QNopt.local_RY, 2)],
        )

        scenario_settings = [
            [np.array([[0], [np.pi]]), np.array([[0], [np.pi]])],
            [np.array([[0, 0]])],
        ]

        priors = [np.ones(2) / 2, np.ones(2) / 2]

        mutual_info = QNopt.mutual_info_cost_fn(ansatz, priors)

        assert np.isclose(mutual_info(scenario_settings), -2)


class TestMutualInfoOptimimzation:
    @pytest.mark.parametrize(
        "priors,match", [([np.ones(3) / 3], 0.9182), ([np.array([0.5, 0.5, 0])], 1)]
    )
    def test_mutual_info_opt_qubit_33(self, priors, match):

        ansatz = QNopt.NetworkAnsatz(
            [QNopt.PrepareNode(3, [0], QNopt.local_RY, 1)],
            [QNopt.MeasureNode(1, 3, [0], QNopt.local_RY, 1)],
        )

        postmap = np.array([[1, 0], [0, 1], [0, 1]])

        mutual_info = QNopt.mutual_info_cost_fn(ansatz, priors, postmap=postmap)

        np.random.seed(12)
        scenario_settings = ansatz.rand_scenario_settings()

        opt_dict = QNopt.gradient_descent(
            mutual_info, scenario_settings, step_size=0.1, num_steps=28, sample_width=24
        )

        assert np.isclose(opt_dict["scores"][-1], match, atol=0.0005)
