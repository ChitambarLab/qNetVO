import pennylane as qml
import pennylane.numpy as np
import itertools


def unitary_matrix(circuit, num_wires, *circ_args, **circ_kwargs):
    """Constructs the unitary matrix representation of a quantum
    circuit in the computational basis.

    :param circuit: A quantum function that accepts no inputs.
    :type circuit: Function

    :param num_wires: The number of wires wires needed by ``circuit``.
    :type num_wires: Int

    :param circ_args: Passthrough arguements for ``circuit``.
    :type circ_args: Positional Arguments

    :param circ_kwargs: Passthrough keyword arguements for ``circuit``.
    :type circ_kwargs: keyword Arguments

    :reurn: A unitary matrix representing the provided ``circuit``.
    :rtype: Numpy Array
    """
    dev = qml.device("default.qubit", wires=range(num_wires))

    @qml.qnode(dev)
    def unitary_z(basis_state):
        qml.BasisState(basis_state, wires=range(num_wires))
        circuit(*circ_args, **circ_kwargs)
        return qml.state()

    bitstrings = list(itertools.product([0, 1], repeat=num_wires))
    u = [unitary_z(bitstring).numpy() for bitstring in bitstrings]
    return np.array(u).T
