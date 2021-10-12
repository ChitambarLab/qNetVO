import pytest
import pennylane as qml
from pennylane import numpy as np
import os
import json

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
    @classmethod
    def construct_opt_dict(self):
        opt_dict = {
            "datetime": "2021-05-22T11:11:11Z",
            "opt_score": 12,
            "opt_settings": [[np.array([[]])], [np.array([[0, 1], [2, 3]]), np.array([[4], [5]])]],
            "scores": [0, 6, 12],
            "samples": [0, 2, 4],
            "step_times": [0.1, 0.01, 0.2],
            "settings_history": [
                [[np.array([[]])], [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]],
                [
                    [np.array([[]])],
                    [np.array([[0, 0.001], [0.002, 0.003]]), np.array([[0.004], [0.005]])],
                ],
                [
                    [np.array([[]])],
                    [np.array([[0, 0.01], [0.02, 0.03]]), np.array([[0.04], [0.05]])],
                ],
                [[np.array([[]])], [np.array([[0, 0.1], [0.2, 0.3]]), np.array([[0.4], [0.5]])]],
                [[np.array([[]])], [np.array([[0, 1], [2, 3]]), np.array([[4], [5]])]],
            ],
        }

        return opt_dict

    @classmethod
    def filename(self):
        return "test/test_write_optimization_json"

    @pytest.fixture()
    def file_io_cleanup(self):
        filepath = self.filename() + ".json"
        if os.path.exists(filepath):
            os.remove(filepath)

        assert not (os.path.exists(filepath))

        yield

        if os.path.exists(filepath):
            os.remove(filepath)

        assert not (os.path.exists(filepath))

    def test_write_optimization_json(self, file_io_cleanup):

        opt_dict = self.construct_opt_dict()

        filename = self.filename()

        assert QNopt.write_optimization_json(opt_dict, filename)

        assert os.path.exists(filename + ".json")

        with open(filename + ".json") as file:
            opt_json = json.load(file)

        assert opt_json["datetime"] == "2021-05-22T11:11:11Z"
        assert opt_json["opt_score"] == 12
        assert opt_json["opt_settings"] == [[[[]]], [[[0, 1], [2, 3]], [[4], [5]]]]
        assert opt_json["scores"] == [0, 6, 12]
        assert opt_json["samples"] == [0, 2, 4]
        assert opt_json["step_times"] == [0.1, 0.01, 0.2]
        assert opt_json["settings_history"] == [
            [[[[]]], [[[0, 0], [0, 0]], [[0], [0]]]],
            [[[[]]], [[[0, 0.001], [0.002, 0.003]], [[0.004], [0.005]]]],
            [[[[]]], [[[0, 0.01], [0.02, 0.03]], [[0.04], [0.05]]]],
            [[[[]]], [[[0, 0.1], [0.2, 0.3]], [[0.4], [0.5]]]],
            [[[[]]], [[[0, 1], [2, 3]], [[4], [5]]]],
        ]

    def test_read_optimization_json(self, file_io_cleanup):

        opt_dict = self.construct_opt_dict()

        filename = self.filename()

        assert QNopt.write_optimization_json(opt_dict, filename)

        assert os.path.exists(filename + ".json")

        opt_json = QNopt.read_optimization_json(filename + ".json")

        assert opt_json["datetime"] == "2021-05-22T11:11:11Z"
        assert opt_json["opt_score"] == 12

        assert len(opt_json["opt_settings"][0]) == 1
        assert len(opt_json["opt_settings"][1]) == 2

        assert isinstance(opt_json["opt_settings"][0][0], np.ndarray)
        assert np.allclose(opt_json["opt_settings"][0][0], np.array([[]]))

        for i in range(2):
            assert isinstance(opt_json["opt_settings"][1][i], np.ndarray)
            assert np.allclose(opt_json["opt_settings"][1][i], opt_dict["opt_settings"][1][i])

        assert opt_json["scores"] == [0, 6, 12]
        assert opt_json["samples"] == [0, 2, 4]
        assert opt_json["step_times"] == [0.1, 0.01, 0.2]

        assert len(opt_json["settings_history"]) == 5

        for i in range(5):
            assert isinstance(opt_json["settings_history"][i][0][0], np.ndarray)
            assert np.allclose(opt_json["settings_history"][i][0][0], np.array([[]]))

            for j in range(2):
                assert isinstance(opt_json["settings_history"][i][1][j], np.ndarray)
                assert np.allclose(
                    opt_json["settings_history"][i][1][j], opt_dict["settings_history"][i][1][j]
                )

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
