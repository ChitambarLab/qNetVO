import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestStatePreparationAnsatzes:
    def test_bell_state_copies(self):
        def bell_state_copies_qnode(wires):
            dev = qml.device("default.qubit", wires=wires)

            @qml.qnode(dev)
            def circuit():
                QNopt.bell_state_copies([], dev.wires)
                return qml.state()

            return circuit

        assert np.allclose(bell_state_copies_qnode([0, 1])(), np.array([1, 0, 0, 1]) / np.sqrt(2))
        assert np.allclose(
            bell_state_copies_qnode([0, 1, 2, 3])(),
            np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) / 2,
        )

    def test_ghz_state(self):
        # setup
        def _ghz_qnode(n_qubits, pauli_obs):
            dev = qml.device("default.qubit", wires=range(n_qubits))

            @qml.qnode(dev)
            def ghz_circuit(settings):
                QNopt.ghz_state(settings, dev.wires)

                obs = pauli_obs(dev.wires[0])
                for wire in dev.wires[1:]:
                    obs = obs @ pauli_obs(wire)

                return qml.expval(obs)

            return ghz_circuit

        # test even n_qubits has perfect correlation
        # odd n_qubits has perfect correlation with expval == 0
        for n_qubits in range(2, 11, 2):
            for pauli_obs in [qml.PauliZ, qml.PauliY, qml.PauliZ]:
                expval = _ghz_qnode(n_qubits, qml.PauliZ)([])
                if n_qubits % 2 == 0:
                    assert np.isclose(expval, 1)
                else:
                    assert np.isclose(expval, 0)

    def test_local_RY(selg):
        qubit2_dev = qml.device("default.qubit", wires=[0, 1])

        @qml.qnode(qubit2_dev)
        def qubit2_RY_circuit(settings):
            QNopt.local_RY(settings, qubit2_dev.wires)
            return qml.state()

        assert np.allclose(qubit2_RY_circuit([np.pi / 2, 0]), np.array([1, 0, 1, 0]) / np.sqrt(2))
        assert np.allclose(qubit2_RY_circuit([0, np.pi / 2]), np.array([1, 1, 0, 0]) / np.sqrt(2))
        assert np.allclose(qubit2_RY_circuit([np.pi / 2, np.pi / 2]), np.array([1, 1, 1, 1]) / 2)

    def test_local_RXRY(self):
        qubit_dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(qubit_dev)
        def qubit_RXRY_circuit(settings):
            QNopt.local_RXRY(settings, qubit_dev.wires)
            return qml.state()

        assert np.allclose(qubit_RXRY_circuit([np.pi / 2, 0]), np.array([1, -1j]) / np.sqrt(2))
        assert np.allclose(qubit_RXRY_circuit([0, np.pi / 2]), np.array([1, 1]) / np.sqrt(2))
        assert np.allclose(qubit_RXRY_circuit([np.pi / 2, np.pi / 2]), [0.5 + 0.5j, 0.5 - 0.5j])

        qubit2_dev = qml.device("default.qubit", wires=[0, 1])

        @qml.qnode(qubit2_dev)
        def qubit2_RXRY_circuit(settings):
            QNopt.local_RXRY(settings, qubit2_dev.wires)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        # x inversion
        assert np.all(qubit2_RXRY_circuit([np.pi, 0, np.pi, 0]) == [-1, -1])

        # y inversion
        assert np.all(qubit2_RXRY_circuit([0, np.pi, 0, np.pi]) == [-1, -1])
