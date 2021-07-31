import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestStatePreparationAnsatzes:
    def test_ghz_state_preparation(self):
        # setup
        def _ghz_qnode(n_qubits, pauli_obs):
            dev = qml.device("default.qubit", wires=range(n_qubits))

            @qml.qnode(dev)
            def ghz_circuit(settings):
                QNopt.ghz_state_preparation(settings, dev.wires)

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
