import pennylane as qml
from pennylane import numpy as np
from pennylane import math


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

    :param static_settings: A set of fixed settings for the preparation node.
    :type static_settings: optional, array-like

    :returns: An instantiated ``PrepareNode`` class.
    """

    def __init__(self, num_in, wires, quantum_fn, num_settings, static_settings=[]):
        super().__init__(wires, quantum_fn)
        self.num_in = num_in
        self.num_settings = num_settings
        self.settings_dims = (num_in, num_settings)
        self.static_settings = static_settings


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

    :param static_settings: A set of fixed settings for the preparation node.
    :type static_settings: optional, array-like

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in, num_out, wires, quantum_fn, num_settings, static_settings=[]):
        super().__init__(num_in, wires, quantum_fn, num_settings, static_settings=static_settings)
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

    :param dev_kwargs: Keyword arguments for the `pennylane.device`_ function.
    :type dev_kwargs: *optional* dict

    .. _pennylane.device: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.device.html?highlight=device#qml-device

    :returns: An instantiated ``NetworkAnsatz`` class with the following fields:

    * **prepare_nodes** - The list of ``PrepareNode`` classes.
    * **measure_nodes** - The list of ``MeasureNode`` classes.
    * **prepare_wires** - The list of wires used by the ``prepare_nodes``.
    * **measure_wires** - The list of wires used by the ``measure_nodes``.
    * **network_wires** - The list of wires used by the network ansatz.
    * **dev_kwargs** - *mutable*, the keyword args to pass to the `pennylane.device`_ function.
                       If no ``dev_kwargs`` are provided, a ``"default.qubit"`` is constructed for
                       noiseless networks and ``"default.mixed"`` device is constructed
                       for noisy networks.
    * **dev** (*qml.device*) - *mutable*, the most recently constructed device for the ansatz.
    * **fn** (*function*) - A quantum function implementing the quantum network ansatz.

    :raises ValueError: If the wires for each ``PrepareNode`` (or ``MeasureNode``) are not unique.
    """

    def __init__(self, prepare_nodes, measure_nodes, noise_nodes=[], dev_kwargs=None):
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

        default_dev_name = "default.qubit" if len(self.noise_nodes) == 0 else "default.mixed"
        self.dev_kwargs = dev_kwargs or {"name": default_dev_name}
        self.dev_kwargs["wires"] = self.network_wires
        self.dev = self.device()

        self.fn = self.construct_ansatz_circuit()

    def set_device(self, name, **kwargs):
        """Configures a new PennyLane device for executing the network ansatz circuit.
        For more details on parameters see `pennylane.device`_.
        This method updates the values stored in ``self.dev_kwargs`` and ``self.dev``.

        :returns: The instantiated device.
        :rtype: ``pennylane.device``
        """

        dev_kwargs = kwargs.copy() if kwargs else {}
        dev_kwargs["name"] = name
        dev_kwargs["wires"] = self.network_wires

        self.dev_kwargs = dev_kwargs
        self.dev = qml.device(**self.dev_kwargs)
        return self.dev

    def device(self):
        """Instantiates a new PennyLane device configured using the ``self.dev_kwargs`` parameters.
        A distinct device is created each time this function runs.

        :returns: The instantiated device.
        :rtype: ``pennylane.device``
        """
        self.dev = qml.device(**self.dev_kwargs)
        return self.dev

    def construct_ansatz_circuit(self):
        prepare_layer = self.circuit_layer(self.prepare_nodes)
        noise_layer = self.circuit_layer(self.noise_nodes)
        measure_layer = self.circuit_layer(self.measure_nodes)

        noise_settings = [np.array([]) for i in range(len(self.noise_nodes))]

        num_prep_settings = math.sum([node.num_settings for node in self.prepare_nodes])

        def ansatz_circuit(settings):
            prep_settings = settings[0:num_prep_settings]
            meas_settings = settings[num_prep_settings:]

            prepare_layer(prep_settings)
            noise_layer(noise_settings)
            measure_layer(meas_settings)

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
    def layer_settings(scenario_settings, node_inputs, nodes):
        """Constructs the list of settings for a circuit layer in the network ansatz.

        The type of tensor used for the returned layer settings matches the tensor
        type of the elements of ``scenario_settings``.

        :param scenario_settings: A list containing the settings for all classical inputs.
        :type network_nodes: list[array[float]]

        :param node_inputs: A list of the classical inputs supplied to each network node.
        :type node_inputs: list[int]

        :returns: A 1D array of all settings for the circuit layer.
        :rtype: array[float]
        """
        return math.concatenate(
            [
                scenario_settings[i][node_input]
                if len(nodes[i].static_settings) == 0
                else nodes[i].static_settings[node_input]
                for i, node_input in enumerate(node_inputs)
            ]
        )

    def qnode_settings(self, scenario_settings, prep_inputs, meas_inputs):
        """Constructs a list of settings to pass to the qnode executing the network ansatz.

        :param scenario_settings: The settings for the network ansatz scenario.
        :type scenario_settings: list[list[np.ndarray]]

        :param prep_inputs: The classical inputs passed to each preparation node.
        :type prep_inputs: list[int]

        :param meas_inputs: The classical inputs passed to each measurement node.
        :type meas_inputs: list[int]

        :returns: The settings to pass to the constructed qnode.
        :rtype: list[float]
        """

        prep_settings = self.layer_settings(scenario_settings[0], prep_inputs, self.prepare_nodes)
        meas_settings = self.layer_settings(scenario_settings[1], meas_inputs, self.measure_nodes)
        return np.concatenate([prep_settings, meas_settings])

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
            2 * np.pi * np.random.random(node.settings_dims) - np.pi
            if len(node.static_settings) == 0
            else np.array([[]])
            for node in self.prepare_nodes
        ]
        measure_settings = [
            2 * np.pi * np.random.random(node.settings_dims) - np.pi
            if len(node.static_settings) == 0
            else np.array([[]])
            for node in self.measure_nodes
        ]

        return [prepare_settings, measure_settings]

    def tf_rand_scenario_settings(self):
        """Creates a randomized settings array for the network ansatz using TensorFlow
        tensor types.

        :returns: See :meth:`qnetvo.NetworkAnsatz.rand_scenario_settings` for details.
        :rtype: list[list[tf.Tensor]]
        """
        from .lazy_tensorflow_import import tensorflow as tf

        np_settings = self.rand_scenario_settings()

        return [
            [tf.Variable(settings) for settings in np_settings[0]],
            [tf.Variable(settings) for settings in np_settings[1]],
        ]

    def zero_scenario_settings(self):
        """Creates a settings array for the network ansatz that consists of zeros.

        :returns: A nested list containing settings for each network node.
                  ``PrepareNode`` settings are listed under index ``0`` while
                  ``MeasureNode`` settings are listed under index ``1``.
                  The measure and prepare layers settings are a list of numpy arrays
                  where the dimension of each array is ``(num_inputs, num_settings)``.
        :rtype: list[list[np.array]]
        """
        prepare_settings = [
            np.zeros(node.settings_dims) if len(node.static_settings) == 0 else np.array([[]])
            for node in self.prepare_nodes
        ]
        measure_settings = [
            np.zeros(node.settings_dims) if len(node.static_settings) == 0 else np.array([[]])
            for node in self.measure_nodes
        ]

        return [prepare_settings, measure_settings]
