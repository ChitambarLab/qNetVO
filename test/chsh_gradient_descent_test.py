import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestCHSHGradientDescent:
    def bell_state_RY(self, settings, wires=[0, 1]):
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=wires)

        qml.RY(settings[0], wires=wires[0])
        qml.RY(settings[1], wires=wires[1])

    def test_chsh_gradient_descent(self):
        prepare_nodes = [
            QNopt.PrepareNode(1, [0, 1], self.bell_state_RY, 2),
        ]
        measure_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)
        chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(
            chsh_cost, init_settings, num_steps=15, step_size=0.2, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-3)

    def test_chsh_gradient_descent_inhomogeneous_settings(self):
        prepare_nodes = [
            QNopt.PrepareNode(1, [0, 1], self.bell_state_RY, 2),
        ]
        measure_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], qml.templates.subroutines.ArbitraryUnitary, 3),
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)
        chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(
            chsh_cost, init_settings, num_steps=30, step_size=0.2, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-3)

    def test_parallel_chsh_gradient_descent(self):
        prepare_nodes = [
            QNopt.PrepareNode(1, [0, 1], self.bell_state_RY, 2),
        ]
        measure_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)
        chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz)
        chsh_grad_fn = QNopt.parallel_chsh_grad(chsh_ansatz)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(
            chsh_cost,
            init_settings,
            num_steps=15,
            step_size=0.2,
            verbose=False,
            grad_fn=chsh_grad_fn,
        )

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-3)

    def test_parallel_chsh_natural_gradient_descent(self):
        prepare_nodes = [
            QNopt.PrepareNode(1, [0, 1], self.bell_state_RY, 2),
        ]
        measure_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 1),
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)
        chsh_cost = QNopt.chsh_inequality_cost(chsh_ansatz)
        chsh_grad_fn = QNopt.chsh_natural_grad(chsh_ansatz)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(
            chsh_cost,
            init_settings,
            num_steps=15,
            step_size=0.1,
            verbose=False,
            grad_fn=chsh_grad_fn,
        )

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-3)
