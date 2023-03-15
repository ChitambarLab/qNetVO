import pytest
import pennylane as qml
import numpy as np
from pennylane import numpy as qnp
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
            "opt_settings": [0, 1, 2, 3, 4, 5],
            "scores": [0, 6, 12],
            "samples": [0, 2, 4],
            "step_times": [0.1, 0.01, 0.2],
            "settings_history": [
                [0, 0, 0, 0, 0, 0],
                [0, 0.001, 0.002, 0.003, 0.004, 0.005],
                [0, 0.01, 0.02, 0.03, 0.04, 0.05],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                [0, 1, 2, 3, 4, 5],
            ],
        }

        return opt_dict

    def construct_opt_dict2(self):
        opt_dict = {
            "datetime": "2021-05-22T11:11:11Z",
            "opt_score": np.array(12),
            "opt_settings": qnp.array([]),
            "scores": [np.array(0), np.array(6), np.array(12)],
            "samples": [0, 2, 4],
            "step_times": [0.1, 0.01, 0.2],
            "settings_history": [qnp.array([]), qnp.array([])],
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
        assert opt_json["opt_settings"] == [0, 1, 2, 3, 4, 5]
        assert opt_json["scores"] == [0, 6, 12]
        assert opt_json["samples"] == [0, 2, 4]
        assert opt_json["step_times"] == [0.1, 0.01, 0.2]
        assert opt_json["settings_history"] == [
            [0, 0, 0, 0, 0, 0],
            [0, 0.001, 0.002, 0.003, 0.004, 0.005],
            [0, 0.01, 0.02, 0.03, 0.04, 0.05],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0, 1, 2, 3, 4, 5],
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

    def test_file_io_integration(self, file_io_cleanup):
        chsh_ansatz = qnetvo.NetworkAnsatz(
            [qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)],
            [
                qnetvo.MeasureNode(
                    num_in=2, num_out=2, wires=[0], ansatz_fn=qnetvo.local_RY, num_settings=1
                ),
                qnetvo.MeasureNode(
                    num_in=2, num_out=2, wires=[1], ansatz_fn=qnetvo.local_RY, num_settings=1
                ),
            ],
        )

        init_settings = chsh_ansatz.rand_network_settings()
        chsh_cost = qnetvo.chsh_inequality_cost_fn(chsh_ansatz)
        opt_dict = qnetvo.gradient_descent(
            chsh_cost, init_settings, num_steps=3, sample_width=1, verbose=False
        )

        qnetvo.write_optimization_json(opt_dict, self.filename())
        opt_json = qnetvo.read_optimization_json(self.filename() + ".json")

        assert opt_dict["opt_settings"] == opt_json["opt_settings"]
        assert opt_dict["settings_history"] == opt_json["settings_history"]
        assert opt_dict["opt_score"] == opt_json["opt_score"]

    def test_read_optimization_json(self, file_io_cleanup):
        opt_dict = self.construct_opt_dict()

        filename = self.filename()

        qnetvo.write_optimization_json(opt_dict, filename)

        assert os.path.exists(filename + ".json")

        opt_json = qnetvo.read_optimization_json(filename + ".json")

        assert opt_json["datetime"] == "2021-05-22T11:11:11Z"
        assert opt_json["opt_score"] == 12

        assert len(opt_json["opt_settings"]) == 6

        assert np.allclose(opt_json["opt_settings"], [0, 1, 2, 3, 4, 5])

        assert opt_json["scores"] == [0, 6, 12]
        assert opt_json["samples"] == [0, 2, 4]
        assert opt_json["step_times"] == [0.1, 0.01, 0.2]

        assert len(opt_json["settings_history"]) == 5

        for i in range(5):
            assert opt_dict["settings_history"][i] == opt_json["settings_history"][i]

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
