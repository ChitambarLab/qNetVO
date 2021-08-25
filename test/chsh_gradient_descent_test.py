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

        @qml.qnode(chsh_ansatz.dev)
        def chsh_circuit(prepare_settings, measure_settings):
            chsh_ansatz.fn(prepare_settings, measure_settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def chsh_cost(scenario_settings):
            score = 0
            prep_settings = chsh_ansatz.layer_settings(scenario_settings[0], [0])
            for x in [0, 1]:
                for y in [0, 1]:
                    meas_settings = chsh_ansatz.layer_settings(scenario_settings[1], [x, y])

                    run = chsh_circuit(prep_settings, meas_settings)
                    scalar = (-1) ** (x * y)

                    score += scalar * run
            return -(score)

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

        @qml.qnode(chsh_ansatz.dev)
        def chsh_circuit(prepare_settings, measure_settings):
            chsh_ansatz.fn(prepare_settings, measure_settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def chsh_cost(scenario_settings):
            score = 0
            prep_settings = chsh_ansatz.layer_settings(scenario_settings[0], [0])

            for x in [0, 1]:
                for y in [0, 1]:
                    meas_settings = chsh_ansatz.layer_settings(scenario_settings[1], [x, y])

                    run = chsh_circuit(prep_settings, meas_settings)
                    scalar = (-1) ** (x * y)

                    score += scalar * run
            return -(score)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(
            chsh_cost, init_settings, num_steps=30, step_size=0.2, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-3)
