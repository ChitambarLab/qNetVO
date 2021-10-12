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


class TestOptimzationFileIO:
    def test_write_optimization_json(self):

        assert False

    def test_read_optimization_json(self):

        assert False

    @pytest.mark.parametrize(
        "prep_settings,meas_settings",
        [([[[]]], [[[]]]), ([[[1, 0]], [[2], [3]]], [[[4, 5], [6, 7]], [[]]])],
    )
    def test_settings_to_np(self, prep_settings, meas_settings):

        settings = [prep_settings, meas_settings]

        np_settings = QNopt.settings_to_np(settings)

        assert isinstance(np_settings, list)
        assert isinstance(np_settings[0], list)
        assert isinstance(np_settings[1], list)

        assert all([isinstance(prep_set, np.ndarray) for prep_set in np_settings[0]])
        assert all([isinstance(meas_set, np.ndarray) for meas_set in np_settings[1]])

        assert all(
            [np.allclose(settings[0][i], np_settings[0][i]) for i in range(len(prep_settings))]
        )
        assert all(
            [np.allclose(settings[1][i], np_settings[1][i]) for i in range(len(meas_settings))]
        )

    @pytest.mark.parametrize(
        "prep_settings,meas_settings",
        [([[[]]], [[[]]]), ([[[1, 0]], [[2], [3]]], [[[4, 5], [6, 7]], [[]]])],
    )
    def test_settings_to_list(self, prep_settings, meas_settings):

        np_settings = [
            [np.array(prep_set) for prep_set in prep_settings],
            [np.array(meas_set) for meas_set in meas_settings],
        ]

        settings = QNopt.settings_to_list(np_settings)

        assert isinstance(settings, list)
        assert isinstance(settings[0], list)
        assert isinstance(settings[1], list)

        assert all([isinstance(prep_set, list) for prep_set in settings[0]])
        assert all([isinstance(meas_set, list) for meas_set in settings[1]])

        assert all(
            [np.allclose(settings[0][i], np_settings[0][i]) for i in range(len(prep_settings))]
        )
        assert all(
            [np.allclose(settings[1][i], np_settings[1][i]) for i in range(len(meas_settings))]
        )
