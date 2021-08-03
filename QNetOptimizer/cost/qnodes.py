import pennylane as qml


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


def local_parity_expval_qnode(network_ansatz):
    """Constructs a qnode that computes expectation values for the local parity observable
    at each measurement node.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz
    """
    observables = local_parity_observables(network_ansatz.measure_nodes)

    @qml.qnode(network_ansatz.dev)
    def circuit(prepare_settings, measure_settings):
        network_ansatz.fn(prepare_settings, measure_settings)

        return [qml.expval(obs) for obs in observables]

    return circuit


def joint_probs_qnode(network_ansatz):
    """Constructs a qnode that computes the joint probabilities in the computational basis
    across all measurement wires.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz
    """

    @qml.qnode(network_ansatz.dev)
    def circuit(prepare_settings, measure_settings):
        network_ansatz.fn(prepare_settings, measure_settings)

        return qml.probs(wires=network_ansatz.measure_wires)

    return circuit
