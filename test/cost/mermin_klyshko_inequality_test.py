import pytest
import pennylane as qml
from pennylane import numpy as np

from context import qnetvo as QNopt


class TestMerminKlyshkoInequality:
    @pytest.mark.parametrize(
        "n, match_inputs, match_scalars",
        [
            (1, [[0]], [1]),
            (2, [[0, 0], [0, 1], [1, 0], [1, 1]], [1, 1, 1, -1]),
            (3, [[0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0]], [2, -2, 2, 2]),
            (
                4,
                [
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [1, 0, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1],
                ],
                [2, 2, 2, -2, -2, -2, -2, 2, 2, 2, 2, -2, 2, 2, 2, -2],
            ),
        ],
    )
    def test_mermin_klyshko_inputs_scalars(self, n, match_inputs, match_scalars):

        meas_inputs_list, scalars_list = QNopt.mermin_klyshko_inputs_scalars(n)

        assert scalars_list == match_scalars

        for i, inputs in enumerate(meas_inputs_list):
            assert inputs == match_inputs[i]

    @pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 10, 11])
    def test_num_terms_mermin_klyshk_inputs_scalars(self, n):
        meas_inputs_list, scalars_list = QNopt.mermin_klyshko_inputs_scalars(n)

        num_terms = 2 ** (2 * np.floor(n / 2))

        assert len(meas_inputs_list) == num_terms
        assert len(scalars_list) == num_terms

    def test_mermin_klyshko_inequality_fn_CHSH_scenario(self):

        prep_nodes = [QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0)]

        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2, 2, [1], QNopt.local_RY, 2),
        ]

        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        mk_chsh_cost = QNopt.mermin_klyshko_cost_fn(ansatz)

        chsh_cost = mk_chsh_cost(
            [
                [np.array([[]])],
                [np.array([[0], [np.pi / 2]]), np.array([[np.pi / 4], [-np.pi / 4]])],
            ]
        )

        assert np.isclose(chsh_cost, -2 * np.sqrt(2))

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10])
    def test_mermin_klyshko_classical_bounds(self, n):
        cl_prep = [QNopt.PrepareNode(1, range(n), lambda settings, wires: None, 0)]

        meas_nodes = [QNopt.MeasureNode(2, 2, [i], QNopt.local_RY, 1) for i in range(n)]

        cl_ansatz = QNopt.NetworkAnsatz(cl_prep, meas_nodes)

        meas_settings = [np.array([[np.pi], [0]]) for i in range(n - 1)]

        if n in [5, 6, 9, 10]:
            meas_settings.append(np.array([[0], [np.pi]]))
        else:
            meas_settings.append(np.array([[np.pi], [0]]))

        cl_opt_settings = [[np.array([[]])], meas_settings]

        mermin_klyshko_cost = QNopt.mermin_klyshko_cost_fn(cl_ansatz)

        assert np.isclose(
            -(mermin_klyshko_cost(cl_opt_settings)), QNopt.mermin_klyshko_classical_bound(n)
        )

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_mermin_klyshko_quantum_bounds(self, n):
        q_prep_ghz = [QNopt.PrepareNode(1, range(n), QNopt.ghz_state, 0)]

        def optimal_meas_ansatz(settings, wires):
            qml.RZ(settings[0], wires=wires)
            qml.RY(np.pi / 2, wires=wires)

        meas_nodes = [QNopt.MeasureNode(2, 2, [i], optimal_meas_ansatz, 1) for i in range(n)]

        q_opt_ansatz = QNopt.NetworkAnsatz(q_prep_ghz, meas_nodes)

        mermin_klyshko_cost = QNopt.mermin_klyshko_cost_fn(q_opt_ansatz)

        opt_meas_settings = [np.array([[-np.pi / 4], [np.pi / 4]]) for i in range(n - 1)]

        if n % 2 == 0:
            opt_meas_settings.append(np.array([[0], [np.pi / 2]]))
        else:
            opt_meas_settings.append(np.array([[np.pi], [3 * np.pi / 2]]))

        opt_settings = [[np.array([[]])], opt_meas_settings]

        assert np.isclose(
            -(mermin_klyshko_cost(opt_settings)), QNopt.mermin_klyshko_quantum_bound(n)
        )
