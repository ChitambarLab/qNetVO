import pennylane as qml
import math


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
    `pennylane.AmplitudeDamping <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AmplitudeDamping.html?highlight=amplitudedamping#pennylane.AmplitudeDamping>`_
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
    `pennylane.PhaseDamping <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.PhaseDamping.html?highlight=phasedamping#pennylane.PhaseDamping>`_
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
