import pennylane as qml
from pennylane import numpy as qnp
from pennylane import math

from .network_nodes import *


class NetworkAnsatz:
    """The ``NetworkAnsatz`` class describes a parameterized quantum network.

    :param layers: Each layer represents a chronological step in the network simulation.
                   The nodes in each layer apply their operations in parallel where no two nodes can
                   operate on the same wire.
    :type layers: list[list[:class:`qnetvo.NetworkNode`]]

    :param dev_kwargs: Keyword arguments for the `pennylane.device`_ function.
    :type dev_kwargs: *optional* dictionary

    .. _pennylane.device: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.device.html

    :returns: An instantiated ``NetworkAnsatz``.

    There are some conventions that should be followed when defining network layers:

    1. The first layer should contain all :class:`qnetvo.PrepareNode` s in the network.
    2. The last layer should contain all :class:`qnetvo.MeasureNode` s in the network.
    3. If classical communication is considered, then a :class:`qnetvo.CCSenderNode` must be used
       to obtain the communicated values in a layer preceding the layers where :class:`qnetvo.CCReceiverNode` s
       consume the classical communication.

    ATTRIBUTES:

    * **layers** - ``list[list[NetworkNode]]``, The input layers of network nodes.
    * **layers_wires** - ``list[qml.wires.Wires]``, The wires used for each layer.
    * **layers_cc_wires_in** - ``list[qml.wires.Wires]``, The classical communication wires input to each network layer.
    * **layers_cc_wires_out** - ``list[qml.wires.Wires]``, The classical communication wires output from each network layer.
    * **layers_num_settings** - ``list[int]``, The number of setting used in each layer.
    * **layers_total_num_in** - ``list[int]``, The total number of inputs for each layer.
    * **layers_node_num_in** - ``list[list[int]]``, The number of inputs for each node in the layer.
    * **layers_num_nodes** - ``list[int]``, The number of nodes in each layer.
    * **network_wires** - The list of wires used by the network ansatz.
    * **network_cc_wires** - The list of classical communication wires in the network.
    * **num_cc_wires** - The number of classical communication wires.
    * **dev_kwargs** - *mutable*, the keyword args to pass to the `pennylane.device`_ function.
                       If no ``dev_kwargs`` are provided, a ``"default.qubit"`` is constructed for
                       noiseless networks.
    * **fn** (*function*) - A quantum function implementing the quantum network ansatz.
    * **parameter_partitions** (*List[List[List[Tuple]]]*) - A ragged array containing tuples that specify
                               how to partition a 1D array of network settings into the subset of settings
                               passed to the qnode simulating the network. See
                               :meth:`get_network_parameter_partitions` for details.

    :raises ValueError: If the ``wires`` are not unique across all nodes in an ansatz layer or the ``cc_wires_out`` are not unique across all layers.
    :raises ValueError: If ``cc_wires_in`` are used by a layer preceding the classical values output onto ``cc_wires_out``.
    """

    def __init__(self, *layers, dev_kwargs=None):
        self.layers = layers

        # layers attributes
        self.layers_wires = [self.collect_wires([node.wires for node in layer]) for layer in layers]
        self.layers_cc_wires_in = [
            self.collect_wires([node.cc_wires_in for node in layer], check_unique=False)
            for layer in layers
        ]
        self.layers_cc_wires_out = [
            self.collect_wires([node.cc_wires_out for node in layer]) for layer in layers
        ]
        self.check_cc_causal_structure(self.layers_cc_wires_in, self.layers_cc_wires_out)

        self.layers_num_settings = [
            math.sum([node.num_settings for node in layer]) for layer in layers
        ]
        self.layers_total_num_in = [math.prod([node.num_in for node in layer]) for layer in layers]
        self.layers_node_num_in = [[node.num_in for node in layer] for layer in self.layers]
        self.layers_num_nodes = [len(layer) for layer in self.layers]

        # network attributes
        self.network_wires = qml.wires.Wires.all_wires(self.layers_wires)
        self.network_cc_wires = self.collect_wires(
            [node.cc_wires_out for layer in layers for node in layer]
        )
        self.num_cc_wires = len(self.network_cc_wires)

        # device attributes
        default_dev_name = "default.qubit"
        self.dev_kwargs = dev_kwargs or {"name": default_dev_name}
        self.dev_kwargs["wires"] = self.network_wires

        # ansatz function attributes
        self.fn = self.ansatz_circuit_fn()
        self.parameter_partitions = self.get_network_parameter_partitions()

    def __call__(self, settings=[]):
        self.fn(settings)

    def ansatz_circuit_fn(self):
        layer_fns = [self.circuit_layer_fn(layer_nodes) for layer_nodes in self.layers]

        def ansatz_circuit(settings=[]):
            cc_wires = [None] * self.num_cc_wires

            start_id = 0
            for i, layer_fn in enumerate(layer_fns):
                end_id = start_id + self.layers_num_settings[i]
                layer_settings = settings[start_id:end_id]
                layer_fn(layer_settings, cc_wires)
                start_id = end_id

        return ansatz_circuit

    @staticmethod
    def circuit_layer_fn(layer_nodes):
        """Constructs a quantum function for an ansatz layer of provided network nodes.

        :param layer_nodes: A list of nodes in a network layer.
        :type layer_nodes: list[:class:`qnetvo.NetworkNode`]

        :returns: A quantum function evaluated as ``circuit_layer(settings)`` where ``settings``
                  is an array constructed via the ``layer_settings`` function.
        :rtype: function
        """

        def circuit_layer(settings, cc_wires):
            start_id = 0
            for node in layer_nodes:
                end_id = start_id + node.num_settings
                node_settings = settings[start_id:end_id]

                node_cc_wires = [cc_wires[i] for i in node.cc_wires_in]

                if isinstance(node, CCSenderNode):
                    cc_out = node(node_settings, node_cc_wires)
                    for i in range(len(cc_out)):
                        cc_wires[node.cc_wires_out[i]] = cc_out[i]
                else:
                    node(node_settings, node_cc_wires)

                start_id = end_id

        return circuit_layer

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
        for i, layer in enumerate(self.layers):
            parameter_partitions += [[]]
            for j, node in enumerate(layer):
                parameter_partitions[i] += [[]]
                for _ in range(node.num_in):
                    stop_id = start_id + node.num_settings
                    parameter_partitions[i][j] += [(start_id, stop_id)]
                    start_id = stop_id

        return parameter_partitions

    @staticmethod
    def collect_wires(wires_lists, check_unique=True):
        """A helper method for the ``NetworkAnsatz`` class which collects and aggregates the
        wires from a set of collection of network nodes (``prepare_nodes`` or ``measure_nodes``).

        :param network_nodes: A list consisting of either ``PrepareNode``'s or ``MeasureNode``'s.
        :type network_nodes: list[PrepareNode or MeasureNode]

        :raises ValueError: If the same wire is used in two different nodes in ``network_nodes``.
        """

        wires_objs = list(map(qml.wires.Wires, wires_lists))
        all_wires = qml.wires.Wires.all_wires(wires_objs)

        if check_unique:
            unique_wires = qml.wires.Wires.unique_wires(wires_objs)

            # two nodes cannot prepare a state on the same wire
            if not all_wires.tolist() == unique_wires.tolist():
                raise ValueError(
                    "One or more wires are not unique. Each node must contain unique wires."
                )

        return all_wires

    @staticmethod
    def check_cc_causal_structure(cc_wires_in_layers, cc_wires_out_layers):
        """Verifies that the classical communication is causal.

        Note that ``cc_wires_out`` describes the classical communication senders while ``cc_wires_in`` describe
        classical communication receivers.
        All network ansatzes must have a causal structure where nodes output their classical communication in
        layers that precede the nodes that use that classical communication.

        :params cc_wires_in_layers: The classical communication input wires, ``cc_wires_in``, considered for each layer.
        :type cc_wires_in_layers: list[qml.wires.Wires]

        :params cc_wires_out_layers: The classical communication output wires, ``cc_wires_out``, considered for each layer.
        :type cc_wires_out_layers: list[qml.wires.Wires]

        :returns: ``True`` if the

        :raises ValueError: If ``cc_wires_in`` are used by a layer preceding the classical values output onto ``cc_wires_out``.
        """
        num_layers = len(cc_wires_in_layers)
        for i in range(num_layers):
            measured_cc_wires = (
                qml.wires.Wires.all_wires([measured_cc_wires, cc_wires_out_layers[i - 1]])
                if (i - 1) >= 0
                else qml.wires.Wires([])
            )

            if not measured_cc_wires.contains_wires(cc_wires_in_layers[i]):
                raise ValueError(
                    "The `cc_wires_in` of layer "
                    + str(i)
                    + " do not have corresponding `cc_wires_out` in a preceding layer."
                )

        return True

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
        """Constructs network settings from qnode settings and the network inputs.

        This implements the reverse mapping implemented in :meth:`qnetvo.NetworkAnsatz.qnode_settings`.
        Since there are fewer qnode settings than network settings, empty elements in the returned
        list are set to zero.

        :param qn_settings: Settings to pass to the network qnode.
        :type qn_settings: array[float]

        :param network_inputs: The considered classical network inputs.
        :type network_inputs: list[int]

        :returns: The network settings
        :rtype: array[float]
        """
        expanded_settings = self.zero_network_settings()

        qn_start_id = 0
        for i, layer_inputs in enumerate(network_inputs):
            for j, node_input in enumerate(layer_inputs):
                node = self.layers[i][j]
                qn_stop_id = qn_start_id + node.num_settings
                start_id, stop_id = self.parameter_partitions[i][j][node_input]

                expanded_settings[start_id:stop_id] = qn_settings[qn_start_id:qn_stop_id]

                qn_start_id = qn_stop_id

        return qml.math.stack(expanded_settings)

    def rand_network_settings(self, fixed_setting_ids=[], fixed_settings=[]):
        """Creates an array of randomized differentiable settings for the network ansatz.
        If fixed settings are specified, then they are marked as ``requires_grad=False`` and
        not differentatiated during optimzation.

        :param fixed_setting_ids: The ids of settings that are held constant during optimization.
                                  Also requires `fixed_settings` to be provided.
        :type fixed_setting_ids: *optional* List[Int]

        :param fixed_settings: The constant values for fixed settings.
        :type fixed_settings: *optional* List[Float]

        :returns: A 1D list of ``qnp.tensor`` scalar values having ``requires_grad=True``.
        :rtype: List[Float]
        """
        num_settings = self.parameter_partitions[-1][-1][-1][-1]
        rand_settings = [
            qnp.array(2 * qnp.pi * qnp.random.rand() - qnp.pi) for _ in range(num_settings)
        ]
        if len(fixed_setting_ids) > 0 and len(fixed_settings) > 0:
            for i, id in enumerate(fixed_setting_ids):
                rand_settings[id] = qnp.array(fixed_settings[i], requires_grad=False)

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
        return [qnp.array(0, requires_grad=True) for _ in range(num_settings)]
