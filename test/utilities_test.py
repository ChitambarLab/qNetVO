import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestUtilities:
    def test_unitary_Matrix(self):
        def circ_pauli_y(wires):
            qml.PauliY(wires=wires)

        U = QNopt.unitary_matrix(circ_pauli_y, 1, wires=[0])
        assert np.allclose(U, qml.PauliY.matrix)

        def circ_rot_y(settings, wires):
            qml.RY(settings, wires=wires)

        theta = np.pi / 4
        U = QNopt.unitary_matrix(circ_rot_y, 1, theta, wires=[0])
        U_match = np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]
        )

        assert np.allclose(U, U_match)

        def circ_cnot():
            qml.CNOT(wires=[0, 1])

        U = QNopt.unitary_matrix(circ_cnot, 2)
        assert np.allclose(U, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])