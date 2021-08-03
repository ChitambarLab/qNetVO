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


def local_parity_observables(measure_nodes):
    """Constructs a list of dichotomic observables for each measurement node.
    The observables are constructed as products of ``qml.PauliZ`` qubit operators and
    therefore, a :math:`+1` outcome corresponds to an *Even* parity bit string whereas a
    :math:`-1` outcome corresponds to an *Odd* parity bit string.

	:param measure_nodes: A list of ``MeasureNode`` classes for which to construct the observables.
	:type measure_nodes: list[ MeasureNode ]
    """
    obs_list = []
    for node in measure_nodes:
        obs = None
        for wire in node.wires:
            if obs == None:
                obs = qml.PauliZ(wire)
            else:
                obs = obs @ qml.PauliZ(wire)

        obs_list.append(obs)

    return obs_list
