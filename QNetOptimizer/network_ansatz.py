import pennylane as qml
from pennylane import numpy as np

import torch


class NoiseNode:
    """A class that configures each noise node in the quantum network.

    :param wires: A list of wires on which the node is defined.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function which accepts as input the
        positional arguments ``(settings, wires)`` where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :returns: An instantiated ``NoiseNode`` class.
    """

    def __init__(self, wires, quantum_fn):
        self.wires = wires
        self.ansatz_fn = quantum_fn
        self.num_settings = 0


class PrepareNode(NoiseNode):
    """A class that configures each preparation node in the quantum network.

    :param num_in: The number of classical inputs for the node.
    :type num_in: int

    :param wires: A list of wires on which the node is defined.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function which accepts as input the
        positional arguments ``(settings, wires)`` where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :param num_settings: The number of settings that the quantum function accepts.
    :type num_settings: int

    :returns: An instantiated ``PrepareNode`` class.
    """

    def __init__(self, num_in, wires, quantum_fn, num_settings):
        super().__init__(wires, quantum_fn)
        self.num_in = num_in
        self.num_settings = num_settings
        self.settings_dims = (num_in, num_settings)


class MeasureNode(PrepareNode):
    """A class that configures each measurement node in the quantum network.

    :param num_in: The number of classical inputs for the node.
    :type num_in: int

    :param num_out: The number of classical outputs for the node.
    :type num_out: int

    :param wires: A list of wires on which the node is defined.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function which accepts as input the
        positional arguments ``(settings, wires)`` where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :param num_settings: The number of settings that the quantum function accepts.
    :type num_settings: int

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in, num_out, wires, quantum_fn, num_settings):
        super().__init__(num_in, wires, quantum_fn, num_settings)
        self.num_out = num_out


class NetworkAnsatz:
    """The ``NetworkAnsatz`` class describes a parameterized quantum prepare and measure network.
    The ansatz is constructed from a prepare layer and a measure layer.
    The prepare layer is a collection of unitaries which prepare quantum states while the
    measure layer is a collection of unitaries which encode the measurement basis.
    These layers are constructed from the ``prepare_nodes`` and ``measure_nodes`` inputs respectively.

    :param prepare_nodes: A list of ``PrepareNode`` classes.
    :type prepare_nodes: list[PrepareNode]

    :param measure_nodes: A list of ``MeasureNode`` classes.
    :type measure_nodes: list[MeasureNode]

    :param noise_nodes: A list of ``NoiseNode`` classes.
    :type noise_nodes: *optional* list[NoiseNode]

    :returns: An instantiated ``NetworkAnsatz`` class with the following fields:

    * **prepare_nodes** - The list of ``PrepareNode`` classes.
    * **measure_nodes** - The list of ``MeasureNode`` classes.
    * **prepare_wires** - The list of wires used by the ``prepare_nodes``.
    * **measure_wires** - The list of wires used by the ``measure_nodes``.
    * **network_wires** - The list of wires used by the network ansatz.
    * **dev** (*qml.device*) - A PennyLane ``"default.qubit"`` device for the network ansatz.
                               If noise nodes are provided, ``"default.mixed"`` is used instead.
    * **fn** (*function*) - A quantum function implementing the quantum network ansatz.

    :raises ValueError: If the wires for each ``PrepareNode`` (or ``MeasureNode``) are not unique.
    """

    def __init__(self, prepare_nodes, measure_nodes, noise_nodes=[]):
        self.prepare_nodes = prepare_nodes
        self.measure_nodes = measure_nodes
        self.noise_nodes = noise_nodes

        self.prepare_wires = self.collect_wires(prepare_nodes)
        self.measure_wires = self.collect_wires(measure_nodes)
        self.noise_wires = (
            qml.wires.Wires([]) if self.noise_nodes == [] else self.collect_wires(noise_nodes)
        )

        self.network_wires = qml.wires.Wires.all_wires(
            [self.prepare_wires, self.measure_wires, self.noise_wires]
        )

        default_dev_name = "default.qubit" if self.noise_nodes == [] else "default.mixed"
        self.dev = qml.device(default_dev_name, wires=self.network_wires)

        self.fn = self.construct_ansatz_circuit()

    def construct_ansatz_circuit(self):
        prepare_layer = self.circuit_layer(self.prepare_nodes)
        noise_layer = self.circuit_layer(self.noise_nodes)
        measure_layer = self.circuit_layer(self.measure_nodes)

        noise_settings = [np.array([]) for i in range(len(self.noise_nodes))]

        def ansatz_circuit(prepare_settings_array, measure_settings_array):
            prepare_layer(prepare_settings_array)
            noise_layer(noise_settings)
            measure_layer(measure_settings_array)

        return ansatz_circuit

    @staticmethod
    def collect_wires(network_nodes):
        """A helper method for the ``NetworkAnsatz`` class which collects and aggregates the
        wires from a set of collection of network nodes (``prepare_nodes`` or ``measure_nodes``).

        :param network_nodes: A list consisting of either ``PrepareNode``'s or ``MeasureNode``'s.
        :type network_nodes: list[PrepareNode or MeasureNode]

        :raises ValueError: If the same wire is used in two different nodes in ``network_nodes``.
        """
        ansatz_wires = list(map(lambda node: qml.wires.Wires(node.wires), network_nodes))
        all_wires = qml.wires.Wires.all_wires(ansatz_wires)
        unique_wires = qml.wires.Wires.unique_wires(ansatz_wires)

        # two nodes cannot prepare a state on the same wire
        if not all_wires.tolist() == unique_wires.tolist():
            raise ValueError(
                "One or more wires are not unique. Each node must contain unique wires."
            )

        return all_wires

    @staticmethod
    def layer_settings(scenario_settings, node_inputs):
        """Constructs the list of settings for a circuit layer in the network ansatz.

        :param scenario_settings: A list containing the settings for all classical inputs.
        :type network_nodes: list[np.array[float]]

        :param node_inputs: A list of the classical inputs supplied to each network node.
        :type node_inputs: list[int]

        :returns: A 1D array of all settings for the circuit layer.
        :rtype: np.array[float]
        """
        settings = np.array([])
        for i in range(len(node_inputs)):
            settings = np.append(settings, scenario_settings[i][node_inputs[i]])

        return settings

    @staticmethod
    def circuit_layer(network_nodes):
        """Constructs a quantum function for an ansatz layer of provided network nodes.

        :param network_nodes: A list of network nodes which can be either
                              ``NoiseNode``, ``PrepareNode``, or ``MeasureNode``
        :type network_nodes: list[NetworkNode]

        :returns: A quantum function evaluated as ``circuit(settings)`` where ``settings``
                  is an array constructed via the ``layer_settings`` function.
        :rtype: function
        """

        def circuit(settings):
            current_id = 0
            for node_id in range(len(network_nodes)):
                node = network_nodes[node_id]
                node_settings = settings[current_id : current_id + node.num_settings]

                current_id += node.num_settings

                node.ansatz_fn(node_settings, node.wires)

        return circuit

    def rand_scenario_settings(self):
        """Creates a randomized settings array for the network ansatz.

        :returns: A nested list containing settings for each network node.
                  ``PreparNode`` settings are listed under index ``0`` while
                  ``MeasureNode`` settings are listed under index ``1``.
                  The measure and prepare layers settings are a list of numpy arrays
                  where the dimension of each array is ``(num_inputs, num_settings)``.
        :rtype: list[list[np.array]]
        """
        prepare_settings = [
            2 * np.pi * np.random.random(node.settings_dims) - np.pi for node in self.prepare_nodes
        ]
        measure_settings = [
            2 * np.pi * np.random.random(node.settings_dims) - np.pi for node in self.measure_nodes
        ]

        return [prepare_settings, measure_settings]

    def zero_scenario_settings(self):
        """Creates a settings array for the network ansatz that consists of zeros.

        :returns: A nested list containing settings for each network node.
                  ``PrepareNode`` settings are listed under index ``0`` while
                  ``MeasureNode`` settings are listed under index ``1``.
                  The measure and prepare layers settings are a list of numpy arrays
                  where the dimension of each array is ``(num_inputs, num_settings)``.
        :rtype: list[list[np.array]]
        """
        prepare_settings = [np.zeros(node.settings_dims) for node in self.prepare_nodes]
        measure_settings = [np.zeros(node.settings_dims) for node in self.measure_nodes]

        return [prepare_settings, measure_settings]
