import pennylane as qml
from pennylane import numpy as np
from pennylane import math


class NoiseNode:
    """A class that configures each noise node in the quantum network.

    :param wires: A list of wires on which the node operates.
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
        self.num_in = 1
        self.cc_wires_in = []

    def fn(self, settings, cc_wires):
        self.ansatz_fn(settings, self.wires)


class ProcessingNode(NoiseNode):
    """A class that configures each processing node in the quantum network.

    :param num_in: The number of classical inputs for the node.
    :type num_in: int

    :param wires: A list of wires on which the node operates.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function which accepts as input the
        positional arguments ``(settings, wires)`` where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :param num_settings: The number of settings that the quantum function accepts.
    :type num_settings: int

    :returns: An instantiated ``ProcessingNode`` class.
    """

    def __init__(self, num_in, wires, quantum_fn, num_settings, cc_wires_in=[]):
        super().__init__(wires, quantum_fn)
        self.num_in = num_in
        self.num_settings = num_settings
        self.cc_wires_in = cc_wires_in
        self.settings_dims = (num_in, num_settings)

    def fn(self, settings, cc_wires):
        args = [settings, self.wires]
        if cc_wires:
            args += [cc_wires]

        return self.ansatz_fn(*args)


class PrepareNode(ProcessingNode):
    """A class that configures each preparation node in the quantum network.

    :param num_in: The number of classical inputs for the node.
    :type num_in: int

    :param wires: A list of wires on which the node operates.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function which accepts as input the
        positional arguments ``(settings, wires)``, where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :param num_settings: The number of settings that the quantum function accepts.
    :type num_settings: int

    :returns: An instantiated ``PrepareNode`` class.
    """

    def __init__(self, num_in, wires, quantum_fn, num_settings, cc_wires_in=[]):
        super().__init__(num_in, wires, quantum_fn, num_settings, cc_wires_in)


class MeasureNode(ProcessingNode):
    """A class that configures each measurement node in the quantum network.

    :param num_in: The number of classical inputs for the node.
    :type num_in: int

    :param num_out: The number of classical outputs for the node.
    :type num_out: int

    :param wires: A list of wires on which the node operates.
    :type wires: array[int]

    :param quantum_fn: A PennyLane quantum function that accepts as input the
        positional arguments ``(settings, wires)`` where settings is an *array[float]*
        of length ``num_settings``.
    :type quantum_fn: function

    :param num_settings: The number of settings that the quantum function accepts.
    :type num_settings: int

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in, num_out, wires, quantum_fn, num_settings, cc_wires_in=[]):
        super().__init__(num_in, wires, quantum_fn, num_settings, cc_wires_in)
        self.num_out = num_out


class CCMeasureNode(ProcessingNode):
    """ """

    def __init__(self, num_in, wires, cc_wires_out, quantum_fn, num_settings, cc_wires_in=[]):
        super().__init__(num_in, wires, quantum_fn, num_settings, cc_wires_in)
        self.cc_wires_out = cc_wires_out


class NetworkAnsatz:
    """The ``NetworkAnsatz`` class describes a parameterized quantum prepare and measure network.
    The ansatz is constructed from a prepare layer and a measure layer.
    The prepare layer is a collection of unitaries which prepare quantum states while the
    measure layer is a collection of unitaries which encode the measurement basis.
    These layers are constructed from the ``prepare_nodes`` and ``measure_nodes`` inputs respectively.

    :param network_layers: Positional arguments each being a list of nodes designating a circuit layer.
                           The first layer must contain ``PrepareNode`` classes while the last layer must
                           contain ``MeasureNode`` classes.
                           Intermediate layers can contain either ``NoiseNode`` or ``ProcessingNode`` classes,
                           but all elements must be the same type.
    :type network_layers: list[list[NetworkNode]]

    :param dev_kwargs: Keyword arguments for the `pennylane.device`_ function.
    :type dev_kwargs: *optional* dict

    .. _pennylane.device: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.device.html?highlight=device#qml-device

    :returns: An instantiated ``NetworkAnsatz`` class with the following fields:

    * **network_layers** - ``list[list[NetworkNode]]``, The input layers of network nodes.
    * **network_layers_wires** - ``list[list[qml.Wires]]``, The wires used for each layer.
    * **network_layers_num_settings** - ``list[int]``, The number of setting used in each layer.
    * **network_layers_total_num_in** - ``list[int]``, The total number of inputs for each layer.
    * **network_layers_node_num_in** - ``list[list[int]]``, The number of inputs for each node in the layer.
    * **network_layers_num_nodes** - ``list[int]``, The number of nodes in each layer.
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
    * **parameter_partitions** (*List[List[List[Tuple]]]*) - A ragged array containing tuples that specify
                               how to partition a 1D array of network settings into the subset of settings
                               passed to the qnode simulating the network. See
                               :meth:`get_network_parameter_partitions` for details.

    :raises ValueError: If the wires for each ``PrepareNode`` (or ``MeasureNode``) are not unique.
    """

    def __init__(self, *network_layers, dev_kwargs=None):
        self.network_layers = network_layers
        self.prepare_nodes = network_layers[0]
        self.measure_nodes = network_layers[-1]

        self.network_layers_wires = [self.collect_wires(layer) for layer in network_layers]
        self.network_layers_num_settings = [
            math.sum([node.num_settings for node in layer]) for layer in network_layers
        ]
        self.network_layers_total_num_in = [
            math.prod([node.num_in for node in layer]) for layer in network_layers
        ]
        self.network_layers_node_num_in = [
            [node.num_in for node in layer] for layer in self.network_layers
        ]
        self.network_layers_num_nodes = [len(layer) for layer in self.network_layers]

        self.prepare_wires = self.network_layers_wires[0]
        self.measure_wires = self.network_layers_wires[-1]
        self.network_wires = qml.wires.Wires.all_wires(self.network_layers_wires)

        self.network_cc_wires = self.collect_wires(
            filter(
                lambda n: isinstance(n, CCMeasureNode),
                [node for layer in network_layers for node in layer],
            ),
            wire_type="classical",
        )
        self.num_cc_wires = len(self.network_cc_wires)

        default_dev_name = "default.qubit"
        self.dev_kwargs = dev_kwargs or {"name": default_dev_name}
        self.dev_kwargs["wires"] = self.network_wires
        self.dev = self.device()

        self.fn = self.ansatz_circuit_fn()
        self.parameter_partitions = self.get_network_parameter_partitions()

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

    def ansatz_circuit_fn(self):
        layer_fns = [self.circuit_layer_fn(layer_nodes) for layer_nodes in self.network_layers]

        def ansatz_circuit(settings):
            cc_wires = [None] * self.num_cc_wires

            start_id = 0
            for i, layer_fn in enumerate(layer_fns):
                end_id = start_id + self.network_layers_num_settings[i]
                layer_settings = settings[start_id:end_id]
                layer_fn(layer_settings, cc_wires)
                start_id = end_id

        return ansatz_circuit

    def get_network_parameter_partitions(self):
        """
        A nested list containing tuples that specify how to partition a 1D array
        of network settings into the subset of settings passed to the qnode simulating
        the network. Each tuple ``(start_id, stop_id)`` is indexed by ``layer_id``,
        ``node_id``, and classical ``input_id`` as
        ``parameter_partitions[layer_id][node_id][input_id] => (start_id, stop_id)``.
        The ``start_id`` and ``stop_id`` describe the slice ``network_settings[start_id:stop_id]``.
        """
        parameter_partitions = []

        start_id = 0
        for i, layer in enumerate(self.network_layers):
            parameter_partitions += [[]]
            for j, node in enumerate(layer):

                parameter_partitions[i] += [[]]
                for _ in range(node.num_in):
                    stop_id = start_id + node.num_settings
                    parameter_partitions[i][j] += [(start_id, stop_id)]
                    start_id = stop_id

        return parameter_partitions

    @staticmethod
    def collect_wires(network_nodes, wire_type="quantum"):
        """A helper method for the ``NetworkAnsatz`` class which collects and aggregates the
        wires from a set of collection of network nodes (``prepare_nodes`` or ``measure_nodes``).

        :param network_nodes: A list consisting of either ``PrepareNode``'s or ``MeasureNode``'s.
        :type network_nodes: list[PrepareNode or MeasureNode]

        :raises ValueError: If the same wire is used in two different nodes in ``network_nodes``.
        """
        ansatz_wires = list(
            map(
                lambda node: qml.wires.Wires(
                    node.wires if wire_type == "quantum" else node.cc_wires_out
                ),
                network_nodes,
            )
        )
        all_wires = qml.wires.Wires.all_wires(ansatz_wires)
        unique_wires = qml.wires.Wires.unique_wires(ansatz_wires)

        # two nodes cannot prepare a state on the same wire
        if not all_wires.tolist() == unique_wires.tolist():
            raise ValueError(
                "One or more wires are not unique. Each node must contain unique wires."
            )

        return all_wires

    def layer_settings(self, network_settings, layer_id, layer_inputs):
        """Constructs the list of settings for a circuit layer in the network ansatz.

        :param network_settings: A list containing the circuit settings for each node
                                 in the network.
        :type network_settings: List[Float]

        :param layer_id: The id for the targeted layer.
        :type layer_id: Int

        :param layer_inputs: A list of the classical inputs supplied to each network node.
        :type layer_inputs: List[Int]

        :returns: A 1D array of settings for the circuit layer.
        :rtype: List[float]
        """
        settings = []
        for j, node_input in enumerate(layer_inputs):
            start_id, stop_id = self.parameter_partitions[layer_id][j][node_input]
            settings += [network_settings[k] for k in range(start_id, stop_id)]

        return settings

    def qnode_settings(self, network_settings, network_inputs):
        """Constructs a list of settings to pass to the qnode executing the network ansatz.

        :param network_settings: The settings for the network ansatz scenario.
        :type network_settings: list[list[np.ndarray]]

        :param network_inputs: The classical inputs passed to each network node.
        :type network_inputs: List[List[int]]

        :returns: A list of settings to pass to the constructed qnode.
        :rtype: np.array
        """
        settings = []
        for i, layer_inputs in enumerate(network_inputs):
            settings += self.layer_settings(network_settings, i, layer_inputs)

        return qml.math.stack(settings)

    def expand_qnode_settings(self, qn_settings, network_inputs):
        """ """
        layers = [self.prepare_nodes, self.measure_nodes]
        expanded_settings = self.zero_network_settings()

        qn_start_id = 0
        for i, layer_inputs in enumerate(network_inputs):
            for j, node_input in enumerate(layer_inputs):
                node = layers[i][j]
                qn_stop_id = qn_start_id + node.num_settings
                start_id, stop_id = self.parameter_partitions[i][j][node_input]

                expanded_settings[start_id:stop_id] = qn_settings[qn_start_id:qn_stop_id]

                qn_start_id = qn_stop_id

        return qml.math.stack(expanded_settings)

    @staticmethod
    def circuit_layer_fn(layer_nodes):
        """Constructs a quantum function for an ansatz layer of provided network nodes.

        :param layer_nodes: A list of nodes in a network layer.
        :type layer_nodes: list[NetworkNode]

        :returns: A quantum function evaluated as ``circuit_layer(settings)`` where ``settings``
                  is an array constructed via the ``layer_settings`` function.
        :rtype: function
        """

        def circuit_layer(settings, cc_wires):
            start_id = 0
            for node in layer_nodes:
                end_id = start_id + node.num_settings
                node_settings = settings[start_id:end_id]

                cc_wires_in = [cc_wires[i] for i in node.cc_wires_in]

                if isinstance(node, CCMeasureNode):
                    cc_out = node.fn(node_settings, cc_wires_in)
                    for i in range(len(cc_out)):
                        cc_wires[node.cc_wires_out[i]] = cc_out[i]
                else:
                    node.fn(node_settings, cc_wires_in)

                start_id = end_id

        return circuit_layer

    def rand_network_settings(self, fixed_setting_ids=[], fixed_settings=[]):
        """Creates an array of randomized differentiable settings for the network ansatz.
        If fixed settings are specified, then they are marked as ``requires_grad=False`` and
        not differentatiated during optimzation.

        :param fixed_setting_ids: The ids of settings that are held constant during optimization.
        Also requires `fixed_settings` to be provided.
        :type fixed_setting_ids: *optional* List[Int]

        :param fixed_settings: The constant values for fixed settings.
        :type fixed_settings: *optional* List[Float]

        :returns: A 1D list of ``np.tensor`` scalar values having ``requires_grad=True``.
        :rtype: List[Float]
        """
        num_settings = self.parameter_partitions[-1][-1][-1][-1]
        rand_settings = [
            np.array(2 * np.pi * np.random.rand() - np.pi) for _ in range(num_settings)
        ]
        if len(fixed_setting_ids) > 0 and len(fixed_settings) > 0:
            for i, id in enumerate(fixed_setting_ids):
                rand_settings[id] = np.array(fixed_settings[i], requires_grad=False)

        return rand_settings

    def tf_rand_network_settings(self, fixed_setting_ids=[], fixed_settings=[]):
        """Creates a randomized settings array for the network ansatz using TensorFlow
        tensor types.

        :param fixed_setting_ids: The ids of settings that are held constant during optimization.
        Also requires `fixed_settings` to be provided.
        :type fixed_setting_ids: *optional* List[Int]

        :param fixed_settings: The constant values for fixed settings.
        :type fixed_settings: *optional* List[Float]

        :returns: A 1D list of ``tf.Variable`` and ``tf.constant`` scalar values.
        :rtype: List[tf.Tensor]
        """
        from .lazy_tensorflow_import import tensorflow as tf

        np_settings = self.rand_network_settings(fixed_setting_ids, fixed_settings)
        return [
            tf.Variable(setting) if qml.math.requires_grad(setting) else tf.constant(setting)
            for setting in np_settings
        ]

    def zero_network_settings(self):
        """Creates a settings array for the network ansatz that consists of zeros.

        :returns: A 1D list of ``np.tensor`` scalar values having ``requires_grad=True``.
        :rtype: List[Float]
        """
        num_settings = self.parameter_partitions[-1][-1][-1][-1]
        return [np.array(0, requires_grad=True) for _ in range(num_settings)]
