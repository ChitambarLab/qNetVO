import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestStatePreparationAnsatzes:
    @pytest.mark.parametrize(
        "settings,match",
        [
            ([0, 0, 0], [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            ([0, np.pi, 0], [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
            ([np.pi, np.pi, 0], [0, -1j / np.sqrt(2), -1j / np.sqrt(2), 0]),
            ([0, np.pi / 2, np.pi], [-0.5j, 0.5j, 0.5j, 0.5j]),
            ([np.pi, 0, 0], [-1j / np.sqrt(2), 0, 0, 1j / np.sqrt(2)]),
        ],
    )
    def test_max_entangled_state(self, settings, match):
        @qml.qnode(qml.device("default.qubit", wires=[0, 1]))
        def test_circ(settings):
            QNopt.max_entangled_state(settings, wires=[0, 1])

            return qml.state()

        assert np.allclose(test_circ(settings), match)

    def test_bell_state_copies(self):
        U = QNopt.unitary_matrix(QNopt.bell_state_copies, 2, [], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]]).T / np.sqrt(2)
        )

        U = QNopt.unitary_matrix(QNopt.bell_state_copies, 4, [], [0, 1, 2, 3])
        assert np.allclose(U[:, 0], np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) / 2)

    def test_ghz_state(self):
        U = QNopt.unitary_matrix(QNopt.ghz_state, 2, [], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]]).T / np.sqrt(2)
        )

        U = QNopt.unitary_matrix(QNopt.ghz_state, 3, [], [0, 1, 2])
        assert np.allclose(
            U,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, -1],
                    [0, 1, 0, 0, 0, 0, -1, 0],
                    [0, 0, 1, 0, 0, -1, 0, 0],
                    [0, 0, 0, 1, -1, 0, 0, 0],
                ]
            ).T
            / np.sqrt(2),
        )

    def test_local_RY(self):
        U = QNopt.unitary_matrix(QNopt.local_RY, 2, [np.pi / 2, 0], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]).T / np.sqrt(2)
        )

        U = QNopt.unitary_matrix(QNopt.local_RY, 2, [0, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]]).T / np.sqrt(2)
        )

        U = QNopt.unitary_matrix(QNopt.local_RY, 2, [np.pi / 2, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]]).T / 2
        )

    def test_local_RXRY(self):
        # single qubit cases
        U = QNopt.unitary_matrix(QNopt.local_RXRY, 1, [np.pi / 2, 0], [0])
        assert np.allclose(U, np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2))

        U = QNopt.unitary_matrix(QNopt.local_RXRY, 1, [0, np.pi / 2], [0])
        assert np.allclose(U, np.array([[1, -1], [1, 1]]) / np.sqrt(2))

        U = QNopt.unitary_matrix(QNopt.local_RXRY, 1, [np.pi / 2, np.pi / 2], [0])
        assert np.allclose(U, np.array([[1 + 1j, -1 - 1j], [1 - 1j, 1 - 1j]]) / 2)

        # two-qubit cases
        U = QNopt.unitary_matrix(QNopt.local_RXRY, 2, [np.pi / 2, 0, np.pi / 2, 0], [0, 1])
        assert np.allclose(
            U,
            np.array([[1, -1j, -1j, -1], [-1j, 1, -1, -1j], [-1j, -1, 1, -1j], [-1, -1j, -1j, 1]]).T
            / 2,
        )

        U = QNopt.unitary_matrix(QNopt.local_RXRY, 2, [0, np.pi / 2, 0, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]]).T / 2
        )
