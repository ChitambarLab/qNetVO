import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


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
            qnet.max_entangled_state(settings, wires=[0, 1])

            return qml.state()

        assert np.allclose(test_circ(settings), match)

    def test_bell_state_copies(self):
        U = qnet.unitary_matrix(qnet.bell_state_copies, 2, [], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]]).T / np.sqrt(2)
        )

        U = qnet.unitary_matrix(qnet.bell_state_copies, 4, [], [0, 1, 2, 3])
        assert np.allclose(U[:, 0], np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) / 2)

    def test_ghz_state(self):
        U = qnet.unitary_matrix(qnet.ghz_state, 2, [], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]]).T / np.sqrt(2)
        )

        U = qnet.unitary_matrix(qnet.ghz_state, 3, [], [0, 1, 2])
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
        U = qnet.unitary_matrix(qnet.local_RY, 2, [np.pi / 2, 0], [0, 1])
        assert np.allclose(
            U, np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]).T / np.sqrt(2)
        )

        U = qnet.unitary_matrix(qnet.local_RY, 2, [0, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]]).T / np.sqrt(2)
        )

        U = qnet.unitary_matrix(qnet.local_RY, 2, [np.pi / 2, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]]).T / 2
        )

    def test_local_RXRY(self):
        # single qubit cases
        U = qnet.unitary_matrix(qnet.local_RXRY, 1, [np.pi / 2, 0], [0])
        assert np.allclose(U, np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2))

        U = qnet.unitary_matrix(qnet.local_RXRY, 1, [0, np.pi / 2], [0])
        assert np.allclose(U, np.array([[1, -1], [1, 1]]) / np.sqrt(2))

        U = qnet.unitary_matrix(qnet.local_RXRY, 1, [np.pi / 2, np.pi / 2], [0])
        assert np.allclose(U, np.array([[1 + 1j, -1 - 1j], [1 - 1j, 1 - 1j]]) / 2)

        # two-qubit cases
        U = qnet.unitary_matrix(qnet.local_RXRY, 2, [np.pi / 2, 0, np.pi / 2, 0], [0, 1])
        assert np.allclose(
            U,
            np.array([[1, -1j, -1j, -1], [-1j, 1, -1, -1j], [-1j, -1, 1, -1j], [-1, -1j, -1j, 1]]).T
            / 2,
        )

        U = qnet.unitary_matrix(qnet.local_RXRY, 2, [0, np.pi / 2, 0, np.pi / 2], [0, 1])
        assert np.allclose(
            U, np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, -1, -1, 1]]).T / 2
        )


class TestNoiseAnsazes:
    @pytest.mark.parametrize(
        "state_prep_fn",
        [
            (lambda: None),
            (lambda: qml.RX(np.pi / 4, wires=[0])),
            (lambda: qml.RY(np.pi / 4, wires=[0])),
            (lambda: qml.RZ(np.pi / 4, wires=[0])),
            (lambda: qml.Hadamard(wires=[0])),
        ],
    )
    def test_pure_amplitude_damping(self, state_prep_fn):
        dev = qml.device("default.qubit", wires=[0, 1])
        dev_mixed = qml.device("default.mixed", wires=[0])

        # verifying state construction
        @qml.qnode(dev)
        def test_state(noise_param):
            state_prep_fn()
            qnet.pure_amplitude_damping([noise_param], wires=[0, 1])

            return qml.state()

        @qml.qnode(dev_mixed)
        def match_state(noise_param):
            state_prep_fn()
            qml.AmplitudeDamping(noise_param, wires=[0])

            return qml.state()

        for noise_param in np.arange(0, 1.001, 1 / 10):
            test_vec = test_state(noise_param)
            AB_state = np.outer(test_vec, test_vec.conj().T)

            A_state = np.zeros((2, 2), dtype=np.complex128)

            A_state[0, 0] = np.trace(AB_state[0:2, 0:2])
            A_state[0, 1] = np.trace(AB_state[0:2, 2:4])
            A_state[1, 0] = np.trace(AB_state[2:4, 0:2])
            A_state[1, 1] = np.trace(AB_state[2:4, 2:4])

            assert np.allclose(A_state, match_state(noise_param))

        # verifying expectation values
        for obs in [qml.PauliX, qml.PauliY, qml.PauliZ]:

            @qml.qnode(dev)
            def test_expval(noise_param):
                state_prep_fn()
                qnet.pure_amplitude_damping([noise_param], wires=[0, 1])

                return qml.expval(obs(wires=[0]))

            @qml.qnode(dev_mixed)
            def match_expval(noise_param):
                state_prep_fn()
                qml.AmplitudeDamping(noise_param, wires=[0])

                return qml.expval(obs(wires=[0]))

            for noise_param in np.arange(0, 1.001, 1 / 10):

                assert np.isclose(test_expval(noise_param), match_expval(noise_param))

    @pytest.mark.parametrize(
        "state_prep_fn",
        [
            (lambda: None),
            (lambda: qml.RX(np.pi / 4, wires=[0])),
            (lambda: qml.RY(np.pi / 4, wires=[0])),
            (lambda: qml.RZ(np.pi / 4, wires=[0])),
            (lambda: qml.Hadamard(wires=[0])),
        ],
    )
    def test_pure_phase_damping(self, state_prep_fn):
        dev = qml.device("default.qubit", wires=[0, 1])
        dev_mixed = qml.device("default.mixed", wires=[0])

        # verifying state construction
        @qml.qnode(dev)
        def test_state(noise_param):
            state_prep_fn()
            qnet.pure_phase_damping([noise_param], wires=[0, 1])

            return qml.state()

        @qml.qnode(dev_mixed)
        def match_state(noise_param):
            state_prep_fn()
            qml.PhaseDamping(noise_param, wires=[0])

            return qml.state()

        for noise_param in np.arange(0, 1.001, 1 / 10):
            test_vec = test_state(noise_param)
            AB_state = np.outer(test_vec, test_vec.conj().T)

            A_state = np.zeros((2, 2), dtype=np.complex128)

            A_state[0, 0] = np.trace(AB_state[0:2, 0:2])
            A_state[0, 1] = np.trace(AB_state[0:2, 2:4])
            A_state[1, 0] = np.trace(AB_state[2:4, 0:2])
            A_state[1, 1] = np.trace(AB_state[2:4, 2:4])

            assert np.allclose(A_state, match_state(noise_param))

        # verifying expectation values
        for obs in [qml.PauliX, qml.PauliY, qml.PauliZ]:

            @qml.qnode(dev)
            def test_expval(noise_param):
                state_prep_fn()
                qnet.pure_phase_damping([noise_param], wires=[0, 1])

                return qml.expval(obs(wires=[0]))

            @qml.qnode(dev_mixed)
            def match_expval(noise_param):
                state_prep_fn()
                qml.PhaseDamping(noise_param, wires=[0])

                return qml.expval(obs(wires=[0]))

            for noise_param in np.arange(0, 1.001, 1 / 10):

                assert np.isclose(test_expval(noise_param), match_expval(noise_param))

    @pytest.mark.parametrize("gamma", np.arange(-0.1, 1.11, 0.1))
    def test_two_qubit_depolarizing(self, gamma):

        dev = qml.device("default.mixed", wires=[0, 1])

        @qml.qnode(dev)
        def test_noise(gamma):
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])

            qnet.two_qubit_depolarizing(gamma, wires=[0, 1])

            return qml.state()

        if not 0 <= gamma <= 1:
            with pytest.raises(
                ValueError,
                match="gamma must be in the interval \\[0,1\\].",
            ):
                test_noise(gamma)
        else:
            test_state = test_noise(gamma)

            bell_state = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
            white_noise_state = np.eye(4) / 4
            match_state = (1 - 16 * gamma / 15) * bell_state + (16 / 15) * gamma * white_noise_state
            print(gamma)
            assert np.allclose(test_state, match_state)

    @pytest.mark.parametrize("gamma", np.arange(-0.1, 1.11, 0.1))
    def test_colored_noise(self, gamma):

        dev = qml.device("default.mixed", wires=[0, 1])

        @qml.qnode(dev)
        def test_noise(gamma):
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])

            qnet.colored_noise(gamma, wires=[0, 1])

            return qml.state()

        if not 0 <= gamma <= 1:
            with pytest.raises(
                ValueError,
                match="gamma must be in the interval \\[0,1\\].",
            ):
                test_noise(gamma)
        else:
            test_state = test_noise(gamma)

            bell_state = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
            colored_noise_state = (
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]) / 2
            )
            match_state = (1 - gamma) * bell_state + gamma * colored_noise_state

            assert np.allclose(test_state, match_state)
