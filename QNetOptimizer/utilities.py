import pennylane as qml
import pennylane.numpy as np
from pennylane import math
import itertools


def unitary_matrix(circuit, num_wires, *circ_args, **circ_kwargs):
    """Constructs the unitary matrix representation of a quantum
    circuit in the computational basis.

    :param circuit: A quantum function.
    :type circuit: Function

    :param num_wires: The number of wires needed by ``circuit``.
    :type num_wires: Int

    :param circ_args: Passthrough arguments for ``circuit``.
    :type circ_args: Positional Arguments

    :param circ_kwargs: Passthrough keyword arguments for ``circuit``.
    :type circ_kwargs: keyword Arguments

    :returns: A unitary matrix representing the provided ``circuit``.
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


def mixed_base_num(n, base_digits):
    """Converts a base-10 number ``n`` into a mixed base number with digit
    values described by the ``base_digits`` array.

    :param n: A base-10 number
    :type n: int

    :param base_digits: A list of integers representing the largest value for each
                        digit in the mixed base number
    :type base_digits: list[int]

    :returns: A list of integers representing the mixed base number.
    :rtype: list[int]

    """
    mixed_base_digits = []
    n_tmp = n
    for i in range(len(base_digits)):
        place = int(math.prod(base_digits[i + 1 :]))

        mixed_base_digits += [n_tmp // place]
        n_tmp = n_tmp % place

    return mixed_base_digits
