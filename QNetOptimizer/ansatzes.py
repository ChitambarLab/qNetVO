import pennylane as qml


def ghz_state_preparation(settings, wires):
    """Initializes a GHZ state on the provided wires.

    :param settings: A placeholder parameter that is not used.
    :type settings: list[empty]

    :param wires: The wires on which the GHZ state is prepared.
    :type wires: qml.Wires
    """
    qml.Hadamard(wires=wires[0])

    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[0], wires[i]])
