import pytest
import pennylane as qml
import numpy as np
import os
import json

import qnetvo


class TestUtilities:
    def test_unitary_matrix(self):
        def circ_pauli_y(wires):
            qml.PauliY(wires=wires)

        U = qnetvo.unitary_matrix(circ_pauli_y, 1, wires=[0])
        assert np.allclose(U, qml.PauliY.compute_matrix())

        def circ_rot_y(settings, wires):
            qml.RY(settings, wires=wires)

        theta = np.pi / 4
        U = qnetvo.unitary_matrix(circ_rot_y, 1, theta, wires=[0])
        U_match = np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]]
        )

        assert np.allclose(U, U_match)

        def circ_cnot():
            qml.CNOT(wires=[0, 1])

        U = qnetvo.unitary_matrix(circ_cnot, 2)
        assert np.allclose(U, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))

    @pytest.mark.parametrize(
        "circuit, num_wires, circ_args, circ_kwargs, basis_state, state_vec_match",
        [
            (qml.Hadamard, 1, (), {"wires": [0]}, np.array([0]), np.array([1, 1]) / np.sqrt(2)),
            (qml.Hadamard, 1, (), {"wires": [0]}, np.array([1]), np.array([1, -1]) / np.sqrt(2)),
            (
                qml.Hadamard,
                2,
                (),
                {"wires": [1]},
                np.array([0, 0]),
                np.array([1, 1, 0, 0]) / np.sqrt(2),
            ),
            (
                qml.RY,
                1,
                (np.pi / 2,),
                {"wires": [0]},
                np.array([1]),
                np.array([-1, 1]) / np.sqrt(2),
            ),
        ],
    )
    def test_state_vec_fn(
        self, circuit, num_wires, circ_args, circ_kwargs, basis_state, state_vec_match
    ):
        state_vec = qnetvo.state_vec_fn(circuit, num_wires)

        assert np.allclose(
            state_vec(*circ_args, basis_state=basis_state, **circ_kwargs),
            state_vec_match,
        )

    @pytest.mark.parametrize(
        "circuit, num_wires, wires_out, circ_args, circ_kwargs, basis_state, density_mat_match",
        [
            (
                qml.Hadamard,
                1,
                [0],
                (),
                {"wires": [0]},
                np.array([0]),
                np.array([[1, 1], [1, 1]]) / 2,
            ),
            (
                qml.Hadamard,
                1,
                [0],
                (),
                {"wires": [0]},
                np.array([1]),
                np.array([[1, -1], [-1, 1]]) / 2,
            ),
            (
                qml.Hadamard,
                2,
                [0],
                (),
                {"wires": [0]},
                np.array([0, 0]),
                np.array([[1, 1], [1, 1]]) / 2,
            ),
            (
                qml.RY,
                1,
                [0],
                (np.pi / 2,),
                {"wires": [0]},
                np.array([1]),
                np.array(
                    [
                        [1, -1],
                        [-1, 1],
                    ]
                )
                / 2,
            ),
            (
                qnetvo.ghz_state,
                2,
                [0],
                ([],),
                {"wires": [0, 1]},
                np.array([0, 0]),
                np.array([[0.5, 0], [0, 0.5]]),
            ),
        ],
    )
    def test_density_mat_fn(
        self, circuit, num_wires, wires_out, circ_args, circ_kwargs, basis_state, density_mat_match
    ):
        density_mat = qnetvo.density_mat_fn(circuit, num_wires)

        assert np.allclose(
            density_mat(wires_out, *circ_args, basis_state=basis_state, **circ_kwargs),
            density_mat_match,
        )


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

    def construct_opt_dict2(self):
        opt_dict = {
            "datetime": "2021-05-22T11:11:11Z",
            "opt_score": np.array(12),
            "opt_settings": [[np.array([[]])], [np.array([[]])]],
            "scores": [np.array(0), np.array(6), np.array(12)],
            "samples": [0, 2, 4],
            "step_times": [0.1, 0.01, 0.2],
            "settings_history": [[[np.array([[]])], [np.array([[]])]]],
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

        qnetvo.write_optimization_json(opt_dict, filename)

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

    def test_write_optimization_json_sanitization(self, file_io_cleanup):
        opt_dict = self.construct_opt_dict2()

        filename = self.filename()

        qnetvo.write_optimization_json(opt_dict, filename)

        assert os.path.exists(filename + ".json")

        with open(filename + ".json") as file:
            opt_json = json.load(file)

        assert opt_json["opt_score"] == 12
        assert opt_json["scores"] == [0, 6, 12]

    def test_read_optimization_json(self, file_io_cleanup):
        opt_dict = self.construct_opt_dict()

        filename = self.filename()

        qnetvo.write_optimization_json(opt_dict, filename)

        assert os.path.exists(filename + ".json")

        opt_json = qnetvo.read_optimization_json(filename + ".json")

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

        np_settings = qnetvo.settings_to_np(settings)

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

        settings = qnetvo.settings_to_list(np_settings)

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

    def test_mixed_base_num(self):
        assert np.all(qnetvo.mixed_base_num(0, [2, 2]) == [0, 0])
        assert np.all(qnetvo.mixed_base_num(2, [2, 2]) == [1, 0])
        assert np.all(qnetvo.mixed_base_num(9, [2, 3, 4]) == [0, 2, 1])
        assert np.all(qnetvo.mixed_base_num(119, [5, 4, 3, 2, 1]) == [4, 3, 2, 1, 0])

    @pytest.mark.parametrize(
        "input, list_dims, match",
        [
            (list(range(10)), [1, 2, 3, 4], [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9]]),
            (list(range(10)), [5, 4, 1], [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]),
        ],
    )
    def test_ragged_reshape(self, input, list_dims, match):
        assert qnetvo.ragged_reshape(input, list_dims) == match

    @pytest.mark.parametrize(
        "input, list_dims",
        [
            (list(range(10)), [2, 3, 4]),
            (list(range(4)), [5, 4, 1]),
        ],
    )
    def test_ragged_reshape_error(self, input, list_dims):
        with pytest.raises(
            ValueError,
            match=r"`len\(input_list\)` must match the sum of `list_dims`\.",
        ):
            qnetvo.ragged_reshape(input, list_dims)
