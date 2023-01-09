class NetworkNode:
    """A quantum network node.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param num_out: The number of classical outputs for the node.
    :type num_out: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param cc_wires_in: The classical communication wires input to the node.
    :type cc_wires_in: list[int]

    :param cc_wires_out: The classical communication wires to output measurement results on.
    :type cc_wires_out: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        called either as ``circuit(settings, wires)``, or as ``circuit(settings, wires, cc_wires)`` where
        ``settings`` is an *array[float]* of length ``num_settings`` and ``cc_wires`` contains the measurement results
        received from upstream nodes.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    :returns: An instance of the ``NetworkNode`` class.

    **Calling a Network Node's Ansatz Function:**

    Given an instantiated network node, ``node = NetworkNode(*args)``, its ansatz quantum circuit functtion
    can be called as ``node(settings, cc_wires)`` or, if ``node.cc_wires_in==[]``, the node can be called
    as  ``node(settings)`` provided that ``cc_wires=[]``.

    **Attributes:**

    All inputs are stored as class attributes under the same name, *e.g.*, the ``num_in`` argument
    is stored as ``node.num_in``.
    """

    def __init__(
        self,
        num_in=1,
        num_out=1,
        wires=[],
        cc_wires_in=[],
        cc_wires_out=[],
        ansatz_fn=None,
        num_settings=0,
    ):
        self.num_in = num_in
        self.num_out = num_out

        self.wires = wires
        self.cc_wires_in = cc_wires_in
        self.cc_wires_out = cc_wires_out

        self.ansatz_fn = ansatz_fn if ansatz_fn else self.default_ansatz_fn
        self.num_settings = num_settings

    def __call__(self, settings=[], cc_wires=[]):
        args = [settings, self.wires]
        if cc_wires:
            args += [cc_wires]

        return self.ansatz_fn(*args)

    def default_ansatz_fn(self, settings, wires, cc_wires=[]):
        pass


class NoiseNode(NetworkNode):
    """A network node that applies noise to its local qubit wires.

    :param wires: The wires on which the node operates.
    :type wires: list[int

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`
        that takes the following form:

        .. code-block:: python

            def noise_ansatz(settings, wires):
                # apply noise operation
                qml.Depolarizing(0.5, wires=wires[0])
                qml.AmplitudeDamping(0.5, wires=wires[1])

        where ``settings=[]`` is unused because noise is considered to be static.

    :type ansatz_fn: function

    We model noise to be independent from classical inputs and upstream measurement results.
    Therefore, only the ``wires`` and ``ansatz_fn`` attributes can be set.
    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    :returns: An instantiated ``NoiseNode`` class.
    """

    def __init__(self, wires=[], ansatz_fn=None):
        super().__init__(wires=wires, ansatz_fn=ansatz_fn)


class ProcessingNode(NetworkNode):
    """A network node that operates upon its local qubit wires where the operation can
    be conditioned upon a classical input or upstream measurement results.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        that takes the following form:

        .. code-block:: python

            def processing_ansatz(settings, wires):
                # apply processing operation
                qml.ArbitraryUnitary(settings[0:16], wires=wires[0:2])

        where ``settings`` is an *array[float]* of length ``num_settings``.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    :returns: An instantiated ``ProcessingNode`` class.
    """

    def __init__(self, num_in=1, wires=[], ansatz_fn=None, num_settings=0):
        super().__init__(
            num_in=num_in,
            wires=wires,
            ansatz_fn=ansatz_fn,
            num_settings=num_settings,
        )


class PrepareNode(ProcessingNode):
    """A network node that initializes a state on its local qubit wires where the
    preparation can be conditioned on a classical input or upstream measurement results.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        that takes the following form:

        .. code-block:: python

            def prepare_ansatz(settings, wires):
                # initalize quantum state from |0...0>
                qml.ArbitraryStatePreparation(settings[0:6], wires=wires[0:2])

        where ``settings`` is an *array[float]* of length ``num_settings``.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    :returns: An instantiated ``PrepareNode`` class.
    """

    pass


class MeasureNode(NetworkNode):
    """A network node that measures its local qubit wires where the measurement can be
    conditioned on a classical input or upstream measurement results.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param num_out: The number of classical outputs for the node.
    :type num_out: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        that takes the following form:

        .. code-block:: python

            def measure_ansatz(settings, wires):
                # rotate measurement basis
                qml.Rot(*settings[0:3], wires=wires[0])

        where ``settings`` is an *array[float]* of length ``num_settings``.
        Note that the measurement ansatz does not apply a measurement operation.
        Measurement operations are specified later when :class:`qnetvo.NetworkAnsatz`
        class is used to construct qnodes and cost functions.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.
    In addition, the number of classical outputs are specified.

    :returns: An instantiated ``MeasureNode`` class.
    """

    def __init__(self, num_in=1, num_out=1, wires=[], ansatz_fn=None, num_settings=0):
        super().__init__(
            num_in=num_in,
            num_out=num_out,
            wires=wires,
            ansatz_fn=ansatz_fn,
            num_settings=num_settings,
        )


class CCSenderNode(NetworkNode):
    """A network node that measures one or more of its local qubits and sends the measurement
    result(s) to one or more downstream :class:`qnetvo.CCReceiverNode` instances.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param cc_wires_out: The classical communication wires to output measurement results on.
    :type cc_wires_out: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        that takes the following form:

        .. code-block:: python

            def cc_sender_ansatz(settings, wires):
                # apply quantum circuit operations
                qml.Rot(*settings[0:3], wires=wires[0])

                # measure qubit to obtain classical communication bit
                cc_bit_out = qml.measure(wires[0])

                # output list of measurement results
                return [cc_bit_out]

        where ``settings`` is an *array[float]* of length ``num_settings``.
        Note that for each wire specified in ``cc_wires_out``, there should be a corresponding
        ``cc_bit_out`` result obtained using `qml.measure`_.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    .. _qml.measure: https://docs.pennylane.ai/en/stable/code/api/pennylane.measure.html

    :returns: An instantiated ``CCSenderNode`` class.
    """

    def __init__(self, num_in=1, wires=[], cc_wires_out=[], ansatz_fn=None, num_settings=0):
        super().__init__(
            num_in=num_in,
            wires=wires,
            ansatz_fn=ansatz_fn,
            num_settings=num_settings,
            cc_wires_out=cc_wires_out,
        )


class CCReceiverNode(NetworkNode):
    """A network node that receives classical communication from an upstream :class:`qnetvo.CCSenderNode`.

    :param num_in: The number of discrete classical inputs that the node accepts.
    :type num_in: int

    :param wires: The wires on which the node operates.
    :type wires: list[int]

    :param cc_wires_in: The classical communication wires input to the node.
    :type cc_wires_in: list[int]

    :param ansatz_fn: A `PennyLane quantum circuit function <https://docs.pennylane.ai/en/stable/introduction/circuits.html>`_
        that takes the following form:

        .. code-block:: python

            def cc_receive_ansatz(settings, wires, cc_wires):
                # apply quantum operations conditioned on classical communication
                qml.cond(cc_wires[0], qml.Rot)(*settings[0:3], wires=wires[0])
                qml.cond(cc_wires[1], qml.Rot)(*settings[3:6], wires=wires[0])

        where ``settings`` is an *array[float]* of length ``num_settings`` and ``cc_wires`` contains
        the measurement results received from upstream nodes.
    :type ansatz_fn: function

    :param num_settings: The number of settings parameterizing the ``ansatz_fn`` circuit.
    :type num_settings: int

    All attributes are inherited from the :class:`qnetvo.NetworkNode` class.

    Note that the classical inputs specified by ``num_in`` are distinct from the classical
    communication inputs passed through ``cc_wires``.
    That is, the classical inputs ``num_in`` are known before the network simulation is run
    whereas the classical communication in ``cc_wires`` are determined during the simulation.

    :returns: An instantiated ``CCReceiverNode`` class.
    """

    def __init__(
        self,
        num_in=1,
        wires=[],
        cc_wires_in=[],
        ansatz_fn=None,
        num_settings=0,
    ):
        super().__init__(
            num_in=num_in,
            wires=wires,
            ansatz_fn=ansatz_fn,
            num_settings=num_settings,
            cc_wires_in=cc_wires_in,
        )
