import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


class TestObservables:
    def test_parity_observable(self):
        qubit_obs = qnet.parity_observable([0])

        assert np.allclose(qubit_obs.matrix(), np.array([[1, 0], [0, -1]]))
        assert qubit_obs.wires == qml.wires.Wires([0])

        qubit2_obs = qnet.parity_observable([0, 1])
        assert np.allclose(qubit2_obs.matrix(), np.diag(np.kron([1, -1], [1, -1])))
        assert qubit2_obs.wires == qml.wires.Wires([0, 1])

        qubit3_obs = qnet.parity_observable([0, 1, 2])
        qubit3_match = np.diag(np.kron([1, -1], np.kron([1, -1], [1, -1])))
        assert np.allclose(qubit3_obs.matrix(), qubit3_match)
        assert qubit3_obs.wires == qml.wires.Wires([0, 1, 2])

    def test_local_parity_observables(self):
        measure_nodes = [
            qnet.MeasureNode(2, 2, range(3 * i, 3 * i + 3), lambda settings, wires: None, 3)
            for i in range(3)
        ]
        observables = qnet.local_parity_observables(measure_nodes)

        obs_matrix_match = np.diag(np.kron([1, -1], np.kron([1, -1], [1, -1])))

        assert np.allclose(observables[0].matrix(), obs_matrix_match)
        assert observables[0].wires == qml.wires.Wires([0, 1, 2])

        assert np.allclose(observables[1].matrix(), obs_matrix_match)
        assert observables[1].wires == qml.wires.Wires([3, 4, 5])

        assert np.allclose(observables[2].matrix(), obs_matrix_match)
        assert observables[2].wires == qml.wires.Wires([6, 7, 8])


class TestQNodes:
    def test_local_parity_expval_qnode(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0, 1], qnet.local_RY, 2),
            qnet.PrepareNode(2, [2, 3], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(1, 2, [0, 1], lambda settings, wires: None, 0),
            qnet.MeasureNode(1, 2, [2, 3], lambda settings, wires: None, 0),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        qnode = qnet.local_parity_expval_qnode(ansatz)

        assert np.all(qnode([0, 0, 0, 0]) == [1, 1])
        assert np.all(qnode([np.pi, 0, 0, 0]) == [-1, 1])
        assert np.all(qnode([np.pi, np.pi, np.pi, 0]) == [1, -1])

    def test_global_parity_expval_qnode(self):
        prep_nodes = [
            qnet.PrepareNode(
                1, [0, 1, 2, 3], lambda settings, wires: qml.BasisState(settings, wires=wires), 4
            )
        ]
        meas_nodes = [
            qnet.MeasureNode(1, 2, [0], lambda settings, wires: None, 0),
            qnet.MeasureNode(1, 2, [1], lambda settings, wires: None, 0),
            qnet.MeasureNode(1, 2, [2], lambda settings, wires: None, 0),
            qnet.MeasureNode(1, 2, [3], lambda settings, wires: None, 0),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        qnode = qnet.global_parity_expval_qnode(ansatz)

        assert qnode(np.array([0, 0, 0, 0])) == 1
        assert qnode(np.array([0, 1, 0, 0])) == -1
        assert qnode(np.array([0, 0, 1, 1])) == 1
        assert qnode(np.array([1, 0, 1, 1])) == -1
        assert qnode(np.array([1, 1, 1, 1])) == 1

    def test_joint_probs_qnode(self):
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(2, [2, 3], qnet.local_RY, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(1, 2, [0], lambda settings, wires: None, 0),
            qnet.MeasureNode(1, 2, [2, 3], lambda settings, wires: None, 0),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        qnode = qnet.joint_probs_qnode(ansatz)

        probs = qnode([0, 0, 0])

        assert len(probs) == 8
        assert np.all(probs == [1, 0, 0, 0, 0, 0, 0, 0])

        assert np.allclose(qnode([np.pi, 0, 0]), [0, 0, 0, 0, 1, 0, 0, 0])
        assert np.allclose(qnode([0, np.pi, 0]), [0, 0, 1, 0, 0, 0, 0, 0])
        assert np.allclose(qnode([0, 0, np.pi]), [0, 1, 0, 0, 0, 0, 0, 0])

    def test_density_matrix_qnode(self):
        prep_nodes = [
            qnet.PrepareNode(1, [0, 1, 2], qnet.W_state, 3),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes)
        qnode = qnet.density_matrix_qnode(ansatz)

        zero_settings = ansatz.zero_network_settings()
        density_matrix = qnode(zero_settings)

        expected_density_matrix = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
                [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.complex128,
        )

        assert np.allclose(
            density_matrix, expected_density_matrix
        ), "Density matrix did not match expected Bell state density matrix."

        qnode_subset_wires = qnet.density_matrix_qnode(ansatz, wires=[0])

        density_matrix_subset = qnode_subset_wires(zero_settings)
        expected_density_matrix_subset = np.array([[2 / 3, 0], [0, 1 / 3]])

        assert np.allclose(
            density_matrix_subset, expected_density_matrix_subset
        ), "Reduced density matrix did not match expected result for wire 0."

        with pytest.raises(
            ValueError, match="Specified wires must be a subset of the wires in the network ansatz."
        ):
            qnet.density_matrix_qnode(ansatz, wires=[3])
