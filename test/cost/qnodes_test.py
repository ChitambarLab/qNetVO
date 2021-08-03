import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestObservables:
    def test_local_parity_observables(self):

        measure_nodes = [
            QNopt.MeasureNode(2, 2, range(3 * i, 3 * i + 3), lambda settings, wires: None, 3)
            for i in range(3)
        ]
        observables = QNopt.local_parity_observables(measure_nodes)

        obs_matrix_match = np.diag(np.kron([1, -1], np.kron([1, -1], [1, -1])))

        assert np.all(observables[0].matrix == obs_matrix_match)
        assert observables[0].wires == qml.wires.Wires([0, 1, 2])

        assert np.all(observables[1].matrix == obs_matrix_match)
        assert observables[1].wires == qml.wires.Wires([3, 4, 5])

        assert np.all(observables[2].matrix == obs_matrix_match)
        assert observables[2].wires == qml.wires.Wires([6, 7, 8])


class TestQNodes:
    def test_local_parity_expval_qnode(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0, 1], QNopt.local_RY, 2),
            QNopt.PrepareNode(2, [2, 3], QNopt.local_RY, 2),
        ]
        meas_nodes = [
            QNopt.MeasureNode(1, 2, [0, 1], lambda settings, wires: None, 0),
            QNopt.MeasureNode(1, 2, [2, 3], lambda settings, wires: None, 0),
        ]

        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        qnode = QNopt.local_parity_expval_qnode(ansatz)

        assert np.all(qnode([[0, 0], [0, 0]], [[], []]) == [1, 1])
        assert np.all(qnode([[np.pi, 0], [0, 0]], [[], []]) == [-1, 1])
        assert np.all(qnode([[np.pi, np.pi], [np.pi, 0]], [[], []]) == [1, -1])

    def test_joint_probs_qnode(self):
        prep_nodes = [
            QNopt.PrepareNode(2, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(2, [2, 3], QNopt.local_RY, 2),
        ]
        meas_nodes = [
            QNopt.MeasureNode(1, 2, [0], lambda settings, wires: None, 0),
            QNopt.MeasureNode(1, 2, [2, 3], lambda settings, wires: None, 0),
        ]

        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)
        qnode = QNopt.joint_probs_qnode(ansatz)

        probs = qnode([[0], [0, 0]], [[], []])

        assert len(probs) == 8
        assert np.all(probs == [1, 0, 0, 0, 0, 0, 0, 0])

        assert np.allclose(qnode([[np.pi], [0, 0]], [[], []]), [0, 0, 0, 0, 1, 0, 0, 0])
        assert np.allclose(qnode([[0], [np.pi, 0]], [[], []]), [0, 0, 1, 0, 0, 0, 0, 0])
        assert np.allclose(qnode([[0], [0, np.pi]], [[], []]), [0, 1, 0, 0, 0, 0, 0, 0])
