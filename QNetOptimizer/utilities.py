import pennylane as qml
import pennylane.numpy as np
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
    dev = qml.device("default.qubit", wires=range(num_wires))

    @qml.qnode(dev)
    def unitary_z(basis_state):
        qml.BasisState(basis_state, wires=range(num_wires))
        circuit(*circ_args, **circ_kwargs)
        return qml.state()

    bitstrings = list(itertools.product([0, 1], repeat=num_wires))
    u = [unitary_z(bitstring).numpy() for bitstring in bitstrings]
    return np.array(u).T


def write_optimization_json(opt_dict, filename):
    """Writest the optimization dictionary to a JSON file.

    :param opt_dict: The dictionary returned by a network optimization.
    :type opt_dict: dict

    :param filename: The name of the JSON file to be written. Note that ``.json`` extension is automatically added.
    :type filename: string

    :returns: ``True``
    """

    opt_dict_json = copy.deepcopy(opt_dict)

    opt_dict_json["opt_settings"] = settings_to_list(opt_dict_json["opt_settings"])
    opt_dict_json["settings_history"] = [
        settings_to_list(settings) for settings in opt_dict_json["settings_history"]
    ]

    with open(filename + ".json", "w") as file:
        file.write(json.dumps(opt_dict_json))

    return True


def read_optimization_json(filepath):
    """Reads data from an optimization JSON created via ``write_optimization_json``.

    :param filepath: The path to the JSON file. Note this string must contain the ``.json`` extension.
    :type filepath: string

    :returns: The optimization dictionary read from the file.
    :rtype: dict
    """

    with open(filepath) as file:
        opt_dict = json.load(file)

    opt_dict["opt_settings"] = settings_to_np(opt_dict["opt_settings"])
    opt_dict["settings_history"] = [
        settings_to_np(settings) for settings in opt_dict["settings_history"]
    ]

    return opt_dict


def settings_to_list(np_scenario_settings):
    """Converts the numpy array in scenario settings to lists.
    This function is intended for printing purposes.

    :param np_scenario_settings: The scenario settings structure for a `NetworkAnsatz`.

    :returns: The same nested array elements and structure using lists.
    """

    list_prep_settings = [node_settings.tolist() for node_settings in np_scenario_settings[0]]
    list_meas_settings = [node_settings.tolist() for node_settings in np_scenario_settings[1]]

    return [list_prep_settings, list_meas_settings]


def settings_to_np(list_scenario_settings):
    """Converts the nested list elements in the scenario settings to numpy arrays.
    This function is intended for printing purposes.

    :param list_scenario_settings: The scenario settings structure for a `NetworkAnsatz`.

    :returns: The same nested array elements and structure using numpy arrays.
    """

    np_prep_settings = [np.array(node_settings) for node_settings in list_scenario_settings[0]]
    np_meas_settings = [np.array(node_settings) for node_settings in list_scenario_settings[1]]

    return [np_prep_settings, np_meas_settings]
