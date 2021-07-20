import pennylane as qml
from pennylane import numpy as np


class PrepareNode:
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
        self.num_in = num_in
        self.wires = wires
        self.ansatz_fn = quantum_fn
        self.num_settings = num_settings
        self.settings_dims = (num_in, num_settings)


class MeasureNode:
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
        self.num_in = num_in
        self.num_out = num_out
        self.wires = wires
        self.ansatz_fn = quantum_fn
        self.num_settings = num_settings
        self.settings_dims = (num_in, num_settings)


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

    :returns: An instantiated ``NetworkAnsatz`` class with the following fields:

    * **prepare_nodes** - The list of ``PrepareNode`` classes.
    * **measure_nodes** - The list of ``MeasureNode`` classes.
    * **prepare_wires** - The list of wires used by the ``prepare_nodes``.
    * **measure_wires** - The list of wires used by the ``measure_nodes``.
    * **network_wires** - The list of wires used by the network ansatz.
    * **dev** (*qml.device*) - A PennyLane ``"default.qubit"`` device for the network ansatz.
    * **fn** (*function*) - A quantum function implementing the quantum network ansatz.

    :raises ValueError: If the wires for each ``PrepareNode`` (or ``MeasureNode``) are not unique.
    """

    def __init__(self, prepare_nodes, measure_nodes):
        self.prepare_nodes = prepare_nodes
        self.measure_nodes = measure_nodes

        self.prepare_wires = self.collect_wires(prepare_nodes)
        self.measure_wires = self.collect_wires(measure_nodes)
        self.network_wires = qml.wires.Wires.all_wires([self.prepare_wires, self.measure_wires])

        self.dev = qml.device("default.qubit", wires=self.network_wires)
        self.fn = self.construct_ansatz_circuit()

    def construct_ansatz_circuit(self):
        prep_layer = self.circuit_layer(self.prepare_nodes)
        meas_layer = self.circuit_layer(self.measure_nodes)

        def ansatz_circuit(prepare_settings_array, measure_settings_array):
            prep_layer(prepare_settings_array)
            meas_layer(measure_settings_array)

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
    def circuit_layer(network_nodes):
        """Constructs a quantum function for an ansatz layer of provided network nodes."""

        def circuit(settings_array):
            for node_id in range(len(network_nodes)):
                node = network_nodes[node_id]
                node.ansatz_fn(settings_array[node_id], node.wires)

        return circuit

    def rand_scenario_settings(self):
        """Creates a randomized settings array for the network ansatz."""
        prepare_settings = [
            2 * np.pi * np.random.random(node.settings_dims) - np.pi for node in self.prepare_nodes
        ]
        measure_settings = [
            2 * np.pi * np.random.random(node.settings_dims) - np.pi for node in self.measure_nodes
        ]

        return [prepare_settings, measure_settings]

    def zero_scenario_settings(self):
        """Creates a settings array for the network ansatz that consists of zeros."""
        prepare_settings = [np.zeros(node.settings_dims) for node in self.prepare_nodes]
        measure_settings = [np.zeros(node.settings_dims) for node in self.measure_nodes]

        return [prepare_settings, measure_settings]
