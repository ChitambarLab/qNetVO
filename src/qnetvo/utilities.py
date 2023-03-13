import pennylane as qml
import pennylane.numpy as qnp
import numpy as np
from pennylane import math
import itertools
import copy
import json


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
    state_vec = state_vec_fn(circuit, num_wires)

    bitstrings = [np.array(bitstring) for bitstring in itertools.product([0, 1], repeat=num_wires)]

    unitary = [
        state_vec(*circ_args, basis_state=bitstring, **circ_kwargs).numpy()
        for bitstring in bitstrings
    ]
    return np.array(unitary).T


def state_vec_fn(circuit, num_wires):
    """Constructs a function ``state_vec(*circ_args, basis_state=[0,...,0], **circ_kwargs)``
    that returns the state vector representation of the output of ``circuit`` where the input
    to the circuit is specified in the computational basis by ``basis_state``.

    :param circuit: A quantum function.
    :type circuit: Function

    :param num_wires: The number of wires to evaluate ``circuit`` on.
    :type num_wires: Int

    :returns: A vector representing the pure quantum state output from ``circuit(*circ_args, **circ_kwargs)``
              when the computational ``basis_state`` is provided as input.
    :rtype: np.array
    """
    dev_wires = range(num_wires)
    dev = qml.device("default.qubit", wires=dev_wires)
    zero_state = np.array([0] * len(dev_wires))

    @qml.qnode(dev)
    def state_vec(*circ_args, basis_state=zero_state, **circ_kwargs):
        qml.BasisState(basis_state, wires=dev_wires)
        circuit(*circ_args, **circ_kwargs)
        return qml.state()

    return state_vec


def density_mat_fn(circuit, num_wires):
    """Constructs a function that returns the density matrix of the specified ``circuit``.

    :param circuit: A quantum function.
    :type circuit: Function

    :param num_wires: The number of wires to evaluate ``circuit`` on.
    :type num_wires: Int

    :returns: A function ``density_mat(wires_out, *circ_args, basis_state=[0,...,0], **circ_kwargs)``  that returns
              the density matrix representing the quantum state on ``wires_out`` for the initialized ``basis_state`` where
              the quantum circuit is called as ``circuit(*circ_args, **circ_kwargs)``.
    :rtype: np.array
    """
    dev_wires = range(num_wires)
    dev = qml.device("default.qubit", wires=dev_wires)
    zero_state = np.array([0] * len(dev_wires))

    @qml.qnode(dev)
    def density_mat(wires_out, *circ_args, basis_state=zero_state, **circ_kwargs):
        qml.BasisState(basis_state, wires=dev_wires)
        circuit(*circ_args, **circ_kwargs)
        return qml.density_matrix(wires=wires_out)

    return density_mat


def write_optimization_json(opt_dict, filename):
    """Writes the optimization dictionary to a JSON file.

    :param opt_dict: The dictionary returned by a network optimization.
    :type opt_dict: dict

    :param filename: The name of the JSON file to be written. Note that ``.json`` extension is automatically added.
    :type filename: string

    :returns: ``None``
    """

    opt_dict_json = copy.deepcopy(opt_dict)

    opt_dict_json["opt_score"] = float(opt_dict_json["opt_score"])
    opt_dict_json["scores"] = [float(score) for score in opt_dict_json["scores"]]
    opt_dict_json["opt_settings"] = [float(setting) for setting in opt_dict_json["opt_settings"]]
    opt_dict_json["settings_history"] = [
        [float(setting) for setting in settings] for settings in opt_dict_json["settings_history"]
    ]

    with open(filename + ".json", "w") as file:
        file.write(json.dumps(opt_dict_json, indent=2))


def read_optimization_json(filepath):
    """Reads data from an optimization JSON created via ``write_optimization_json``.

    :param filepath: The path to the JSON file. Note this string must contain the ``.json`` extension.
    :type filepath: string

    :returns: The optimization dictionary read from the file.
    :rtype: dict
    """

    with open(filepath) as file:
        opt_dict = json.load(file)

    return opt_dict


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


def ragged_reshape(input_list, list_dims):
    """Takes a 1D list ``input_list`` and breaks it into smaller lists having lengths specified by the
    elements of ``list_dims``.

    :param input_list: The list to reshape.
    :type input_list: list

    :param list_dims: The length of each element in the output list
    :type list_dims: list[int]

    :returns: The original list reshaped as a list of lists where each element has length specified
        by ``list_dims``.
    :rtype: list[list]

    :raises ValueError: If `len(input_list) != sum(list_dims)` because list cannot be repartitioned.
    """
    if math.sum(list_dims) != len(input_list):
        raise ValueError("`len(input_list)` must match the sum of `list_dims`.")

    output_list = []
    start_id = 0
    for i, num_nodes in enumerate(list_dims):
        output_list += [[]]
        end_id = start_id + num_nodes
        output_list[i] += input_list[start_id:end_id]
        start_id = end_id

    return output_list
