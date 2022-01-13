import pennylane as qml


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
