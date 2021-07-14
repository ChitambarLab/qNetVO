import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestCHSHGradientDescent:
    def bell_state_RY(self, settings, wires=[0, 1]):
        print("bell_state_RY settings : ", settings)
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=wires)

        qml.RY(settings[0], wires=wires[0])
        qml.RY(settings[1], wires=wires[1])

    def local_RY(self, settings, wires=[0]):
        print("local_RY settings : ", settings)

        qml.RY(settings[0], wires=wires[0])

    def test_chsh_gradient_descent(self):
        prepare_nodes = [
            QNopt.PrepareNode(1, [0, 1], self.bell_state_RY, 2),
        ]
        measure_nodes = [
            QNopt.MeasureNode(2, 2, [0], self.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], self.local_RY, 1),
        ]

        chsh_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)

        @qml.qnode(chsh_ansatz.dev)
        def chsh_circuit(prepare_settings, measure_settings):
            chsh_ansatz.fn(prepare_settings, measure_settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        def chsh_cost(scenario_settings):
            score = 0
            for x in [0, 1]:
                for y in [0, 1]:
                    prep_settings = scenario_settings[0][0]
                    A_meas_settings = scenario_settings[1][0][x]
                    B_meas_settings = scenario_settings[1][1][y]

                    run = chsh_circuit(prep_settings, [A_meas_settings, B_meas_settings])
                    scalar = (-1) ** (x * y)

                    score += scalar * run
            return -(score)

        np.random.seed(666)
        init_settings = chsh_ansatz.rand_scenario_settings()
        opt_dict = QNopt.gradient_descent(chsh_cost, init_settings, verbose=False)

        assert np.isclose(opt_dict["opt_score"], 2 * np.sqrt(2), atol=1e-5)
