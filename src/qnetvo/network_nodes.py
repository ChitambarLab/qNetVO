class NetworkNode:
    """A quantum network node.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        called either as ``circuit(settings, wires)``, or as ``circuit(settings, wires, cc_wires)`` where
        ``settings`` is an *array[float]* of length ``num_settings`` and ``cc_wires`` contains the measurement results
        received from upstream nodes.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    :param cc_wires_in: The classical communication wires input to the node.
    :type cc_wires_in: list[int]

    :returns: An instance of the ``NetworkNode`` class.

    **Calling a Network Node's Ansatz Function:**

    Given an instantiated network node, ``node = NetworkNode(*args)``, its ansatz quantum circuit functtion
    can be called as ``node(settings, cc_wires)`` or, if ``node.cc_wires_in==[]``, the node can be called
    as  ``node(settings)`` if ``cc_wires_in=[]``.

    **Attributes:**

    All inputs are stored as class attributes under the same name, *e.g.*, the ``num_in`` argument
    is stored as ``node.num_in``.
    """

    def __init__(self, num_in, wires, ansatz_fn, num_settings, cc_wires_in=[]):
        self.wires = wires
        self.num_in = num_in
        self.ansatz_fn = ansatz_fn
        self.num_settings = num_settings
        self.cc_wires_in = cc_wires_in

    def __call__(self, settings, cc_wires=[]):
        args = [settings, self.wires]
        if cc_wires:
            args += [cc_wires]

        return self.ansatz_fn(*args)


class NoiseNode(NetworkNode):
    """A network node that applies noise to its local qubit wires.

    All inputs and attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    We model noise to be independent from classical inputs and upstream measurement results.
    Thus, the following class attributes are set as:

    * ``noise_node.num_in == 1``
    * ``noise_node.num_settings == 0``
    * ``noise_node.cc_wires_in == []``

    :returns: An instantiated ``NoiseNode`` class.
    """

    def __init__(self, wires, ansatz_fn):
        super().__init__(num_in=1, wires=wires, ansatz_fn=ansatz_fn, num_settings=0, cc_wires_in=[])


class ProcessingNode(NetworkNode):
    """A network node that operates upon its local qubit wires where the operation can
    be conditioned upon a classical input or upstream measurement results.

    All inputs and attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    :returns: An instantiated ``ProcessingNode`` class.
    """

    pass


class PrepareNode(ProcessingNode):
    """A network node that initializes a state on its local qubit wires where the
    preparation can be conditioned on a classical input or upstream measurement results.

    All inputs and attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    :returns: An instantiated ``PrepareNode`` class.
    """

    pass


class MeasureNode(NetworkNode):
    """A network node that measures its local qubit wires where the measurement can be
    conditioned on a classical input or upstream measurement results.

    All inputs and attributes are inherited from the :class:`qnetvo.NetworkNode` class.
    In addition, the number of classical outputs are specified.

    :param num_out: The number of classical outputs for the node.
    :type num_out: int

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in, num_out, wires, ansatz_fn, num_settings, cc_wires_in=[]):
        super().__init__(num_in, wires, ansatz_fn, num_settings, cc_wires_in)
        self.num_out = num_out


class CCMeasureNode(NetworkNode):
    """A network node that measures its local qubit wires where the measurement can be
    conditioned on a classical input or upstream measurement results.

    All inputs and attributes are inherited from the :class:`qnetvo.NetworkNode` class.
    In addition, the classical communication output wires are specified.
    These wires store the results of mid-circuit measurements.

    :param cc_wires_out: The classical communication wires to output measurement results on.
    :type cc_wires_out: list[int]

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in, wires, cc_wires_out, ansatz_fn, num_settings, cc_wires_in=[]):
        super().__init__(num_in, wires, ansatz_fn, num_settings, cc_wires_in)
        self.cc_wires_out = cc_wires_out
