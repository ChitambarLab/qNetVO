import pennylane as qml
from pennylane.operation import Channel
from pennylane import numpy as np
import math

eps = 1e-7  # constant


def max_entangled_state(settings, wires):
    """Ansatz function for maximally entangled two-qubit state preparation.

    A general ``qml.Rot`` unitary is applied to one-side of a
    Bell state creating a general parameterization of all maximally
    entangled states.

    param settings: A list with three elements parameterizing a general single-qubit unitary
    :type settings: list[float]

    :param wires: The two wires on which the maximally entangled state is prepared.
    :type wires: qml.Wires
    """
    # prepare bell state
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=wires[0:2])

    # perform general rotation on first qubit
    qml.Rot(*settings, wires=wires[0])


def bell_state_copies(settings, wires):
    """Initializes :math:`n` Bell states on :math:`2n` wires.
    The first :math:`n` wires represent Alice's half of the entangled
    state while Bob's half consists of the remaining :math:`n` wires.
    Entanglement is shared between Alice and Bob, however, the independent
    Bell states are not initially entangled with each other.

    :param settings: A placeholder parameter that is not used.
    :type settings: list[empty]

    :param wires: The wires on which the bell states are prepared.
    :type wires: qml.Wires
    """
    for i in range(len(wires) // 2):
        qml.Hadamard(wires=wires[i])
        qml.CNOT(wires=[wires[i], wires[i + len(wires) // 2]])


def ghz_state(settings, wires):
    """Initializes a GHZ state on the provided wires.

    :param settings: A placeholder parameter that is not used.
    :type settings: list[empty]

    :param wires: The wires on which the GHZ state is prepared.
    :type wires: qml.Wires
    """
    qml.Hadamard(wires=wires[0])

    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[0], wires[i]])


def local_RY(settings, wires):
    """Performs a rotation about :math:`y`-axis on each qubit
    specified by ``wires``.

    :param settings: A list of ``len(wires)`` real values.
    :type settings: list[float]

    :param wires: The wires on which the rotations are applied.
    :type wires: qml.Wires
    """
    qml.broadcast(qml.RY, wires, "single", settings)


def local_RXRY(settings, wires):
    """Performs a rotation about the :math:`x` and :math:`y` axes on
    each qubit specified by ``wires``.

    :param settings: A list of ``2*len(wires)`` real values.
    :type settings: list[float]

    :param wires: The wires on which the rotations are applied.
    :type wires: qml.Wires
    """
    for i, wire in enumerate(wires):
        qml.RX(settings[2 * i], wires=wire)
        qml.RY(settings[2 * i + 1], wires=wire)


def pure_amplitude_damping(noise_params, wires):
    """Implements a single qubit amplitude damping channel as a two-qubit gate.

    This allows amplitude damping noise to be applied on quantum hardware or on a
    state-vector simulator such as ``"default.qubit"``.

    This method is equivalent to the
    `pennylane.AmplitudeDamping <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AmplitudeDamping.html>`_
    method. The corresponding Kraus operators are expressed as:

    .. math::

        K_0 = \\begin{pmatrix}1 & 0 \\\\ 0 & \\sqrt{1-\\gamma} \\end{pmatrix}, \\quad
        K_1 = \\begin{pmatrix}0 & \\sqrt{\\gamma} \\\\ 0 & 0 \\end{pmatrix}.

    :param noise_params: The parameter ``[gamma]`` quantifying the amount of amplitude damping noise.
    :type noise_params: List[Float]

    :param wires: Two wires on which to implement the amplitude damping channel.
                  The channel input and output is on ``wires[0]`` while ``wires[1]``
                  serves as the ancilla register.
    :type wires: qml.Wires
    """

    ry_setting = 2 * math.asin(math.sqrt(noise_params[0]))

    qml.ctrl(qml.RY, control=wires[0])(ry_setting, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])


def pure_phase_damping(noise_params, wires):
    """Implements a single qubit phase damping channel as a two-qubit gate.

    This allows phase damping noise to be applied on quantum hardware or on a
    state-vector simulator such as ``"default.qubit"``.

    This method is equivalent to the
    `pennylane.PhaseDamping <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.PhaseDamping.html>`_
    method. The corresponding Kraus operators are expressed as:

    .. math::

        K_0 = \\begin{pmatrix}1 & 0 \\\\ 0 & \\sqrt{1-\\gamma} \\end{pmatrix}, \\quad
        K_1 = \\begin{pmatrix}0 & 0 \\\\ 0 & \\sqrt{\\gamma} \\end{pmatrix}.

    :param noise_params: The parameter ``[gamma]`` quantifying the amount of phase damping noise.
    :type noise_params: List[Float]

    :param wires: Two wires on which to implement the phase damping channel.
                  The channel input and output is on ``wires[0]`` while ``wires[1]``
                  serves as the ancilla register.
    :type wires: qml.Wires
    """

    ry_setting = 2 * math.asin(math.sqrt(noise_params[0]))
    qml.ctrl(qml.RY, control=wires[0])(ry_setting, wires=wires[1])


class two_qubit_depolarizing(Channel):
    """Applies a two-qubit depolarizing channel using Kraus operators.

    The channel is called as a quantum function ``two_qubit_depolarizing(gamma, wires)``

    :param gamma: The amount of depolarizing noise in the channel.
    :type gamma: Float

    :param wires: Two wires on which to apply the depolarizing noise.
    :type wires: qml.Wires

    For a noise parameter :math:`\\gamma`, the two-qubit depolarizing
    noise model is represented by the following function:

    .. math::

        \\mathcal{N}(\\rho) = (1-\\frac{16}{15}\\gamma)\\rho + (\\frac{16}{15})(\\frac{\\gamma}{4})) \\mathbb{I}

    :raises ValueError: if ``0 <= gamma <= 1`` is not satisfied.
    """

    num_params = 1
    num_wires = 2
    grad_method = "F"

    @staticmethod
    def compute_kraus_matrices(gamma):
        """Kraus matrices representing the ``two_qubit_depolarizing`` channel.

        :param gamma: The amount of depolarizing noise in the channel.
        :type gamma: Float

        :returns: The Kraus matrices.
        :rtype: List[Array]
        """

        if not 0.0 - eps <= gamma <= 1.0 + eps:
            raise ValueError("gamma must be in the interval [0,1].")
        elif np.isclose(gamma, 0):
            gamma = 0

        paulis = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]

        kraus_ops = []
        for i in range(4):
            for j in range(4):
                scalar = np.sqrt(1 - gamma) if i + j == 0 else np.sqrt(gamma / 15)
                kraus_ops.append(scalar * np.kron(paulis[i], paulis[j]))

        return kraus_ops


class colored_noise(Channel):
    """Applies a two-qubit colored noise channel using Kraus operators.

    The channel is called as a quantum function ``colored_noise(gamma, wires)``.

    :param gamma: The amount of colored noise in the channel.
    :type gamma: Float

    :param wires: Two wires on which to apply the colored noise.
    :type wires: qml.Wires

    For the noise parameter :math:`\\gamma`, the colored noise model is
    represented by the following function:

    .. math::

        \\mathcal{N}(\\rho) = (1-\\gamma)\\rho + \\frac{\\gamma}{2}(|01\\rangle\\langle 01| + |10\\rangle\\langle 10|)

    :raises ValueError: if ``0 <= gamma <= 1`` is not satisfied.
    """

    num_params = 1
    num_wires = 2
    grad_method = "F"

    @staticmethod
    def compute_kraus_matrices(gamma):
        """Kraus matrices representing the ``colored_noise`` channel.

        :param gamma: The amount of colored noise in the channel.
        :type gamma: Float

        :returns: The Kraus matrices.
        :rtype: List[Array]
        """
        if not 0.0 - eps <= gamma <= 1.0 + eps:
            raise ValueError("gamma must be in the interval [0,1].")
        elif np.isclose(gamma, 0):
            gamma = 0

        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_min = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_min = np.array([0, 1, -1, 0]) / np.sqrt(2)

        scalar = np.sqrt(gamma / 2)

        K0 = np.sqrt(1 - gamma) * np.eye(4)
        K1 = scalar * np.outer(psi_plus, phi_plus)
        K2 = scalar * np.outer(psi_plus, phi_min)
        K3 = scalar * np.outer(psi_plus, psi_plus)
        K4 = scalar * np.outer(psi_plus, psi_min)
        K5 = scalar * np.outer(psi_min, phi_plus)
        K6 = scalar * np.outer(psi_min, phi_min)
        K7 = scalar * np.outer(psi_min, psi_plus)
        K8 = scalar * np.outer(psi_min, psi_min)
        return [K0, K1, K2, K3, K4, K5, K6, K7, K8]
