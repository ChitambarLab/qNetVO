import pennylane as qml
from pennylane import math


def parity_observable(wires):
    """Constructs the parity observable for the provided ``wires`` in the computational basis.

    :param wires: A list of ``MeasureNode`` classes for which to construct the observables.
    :type wires: qml.wires.Wires

    :returns: A parity ``qml.Observable`` across all wires.
    """
    obs = qml.PauliZ(wires=wires[0])
    for wire in wires[1:]:
        obs = obs @ qml.PauliZ(wires=wire)

    return obs


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
        obs_list.append(parity_observable(node.wires))

    return obs_list


def local_parity_expval_qnode(network_ansatz, **qnode_kwargs):
    """Constructs a qnode that computes expectation values for the local parity observable
    at each measurement node.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :returns: A qnode that performs a local parity measurement at measurement nodes.
              The qnode is called as ``qnode(settings)``.
    :rtype: ``qml.QNode``
    """
    observables = local_parity_observables(network_ansatz.layers[-1])

    @qml.qnode(qml.device(**network_ansatz.dev_kwargs), **qnode_kwargs)
    def circuit(settings):
        network_ansatz.fn(settings)

        return [qml.expval(obs) for obs in observables]

    return circuit


def global_parity_expval_qnode(network_ansatz, **qnode_kwargs):
    """Constructs a qnode that computes expectation values for the local parity observable
    at each measurement node.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :returns: A qnode the performs a global parity measurement and is called as ``qnode(settings)``.
    :rtype: ``qml.QNode``
    """
    parity_obs = parity_observable(network_ansatz.layers_wires[-1])

    @qml.qnode(qml.device(**network_ansatz.dev_kwargs), **qnode_kwargs)
    def circuit(settings):
        network_ansatz.fn(settings)

        return qml.expval(parity_obs)

    return circuit


def joint_probs_qnode(network_ansatz, **qnode_kwargs):
    """Constructs a qnode that computes the joint probabilities in the computational basis
    across all measurement wires.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :returns: A qnode called as ``qnode(settings)`` for evaluating the joint probabilities of the
              network ansatz.
    :rtype: ``pennylane.QNode``
    """

    @qml.qnode(qml.device(**network_ansatz.dev_kwargs), **qnode_kwargs)
    def circuit(settings):
        network_ansatz.fn(settings)

        return qml.probs(wires=network_ansatz.layers_wires[-1])

    return circuit
