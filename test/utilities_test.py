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

    def test_mixed_base_num(self):

        assert np.all(QNopt.mixed_base_num(0, [2, 2]) == [0, 0])
        assert np.all(QNopt.mixed_base_num(2, [2, 2]) == [1, 0])
        assert np.all(QNopt.mixed_base_num(9, [2, 3, 4]) == [0, 2, 1])
        assert np.all(QNopt.mixed_base_num(119, [5, 4, 3, 2, 1]) == [4, 3, 2, 1, 0])
