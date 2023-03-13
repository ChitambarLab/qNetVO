import pennylane as qml
from pennylane import math
from ..qnodes import global_parity_expval_qnode
from scipy.linalg import pinvh


def chain_I22_fn(network_ansatz, parallel=False, **qnode_kwargs):
    """Constructs a function for evaluating the :math:`I_{22}` quantity used in the ``nlocal_chain_22_cost_fn`` function.

    :param network_ansatz: The ansatz for the :math:`n`-local chain network.
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True``, remote qnode executions are made in parallel web requests.
    :type parallel: *optional* bool, default ``False``

    :param qnode_kwargs: keyword args to be passed to constructed QNodes.
    :type: *optional* dictionary

    :returns: A function callable as ``I22(*network_settings)`` that evaluates the :math:`I_{22}` quantity.
    :rtype: function
    """

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]

    if parallel:
        from ..lazy_dask_import import dask

        chain_qnodes = [
            global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(4)
        ]
    else:
        chain_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    num_interior_nodes = len(network_ansatz.layers[-1]) - 2

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]
    I22_xy_inputs = [[x_a] + [0 for i in range(num_interior_nodes)] + [x_b] for x_a, x_b in xy_vals]

    def I22(*network_settings):
        I22_xy_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in I22_xy_inputs
        ]

        if parallel:
            I22_delayed_results = [
                dask.delayed(chain_qnodes[i])(settings)
                for i, settings in enumerate(I22_xy_settings)
            ]

            I22_results = math.stack(dask.compute(*I22_delayed_results, scheduler="threads"))
        else:
            I22_results = math.stack([chain_qnode(settings) for settings in I22_xy_settings])

        return math.sum(I22_results)

    return I22


def chain_J22_fn(network_ansatz, parallel=False, **qnode_kwargs):
    """Constructs a function for evaluating the :math:`J_{22}` quantity used in the ``nlocal_chain_22_cost_fn`` function.

    :param network_ansatz: The ansatz for the :math:`n`-local chain network.
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True``, remote qnode executions are made in parallel web requests.
    :type parallel: *optional* bool, default ``False``

    :param qnode_kwargs: keyword args to be passed to constructed QNodes.
    :type: *optional* dictionary

    :returns: A function callable as ``J22(*network_settings)`` that evaluates the :math:`J_{22}` quantity.
    :rtype: function
    """

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]

    if parallel:
        from ..lazy_dask_import import dask

        chain_qnodes = [
            global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(4)
        ]
    else:
        chain_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    num_interior_nodes = len(network_ansatz.layers[-1]) - 2
    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]

    J22_xy_inputs = [[x_a] + [1 for i in range(num_interior_nodes)] + [x_b] for x_a, x_b in xy_vals]

    def J22(*network_settings):
        J22_xy_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in J22_xy_inputs
        ]

        if parallel:
            J22_delayed_results = [
                dask.delayed(chain_qnodes[i])(settings)
                for i, settings in enumerate(J22_xy_settings)
            ]

            J22_results = math.stack(dask.compute(*J22_delayed_results, scheduler="threads"))
        else:
            J22_results = math.stack([chain_qnode(settings) for settings in J22_xy_settings])

        return math.sum(math.stack([1, -1, -1, 1]) * J22_results)

    return J22


def nlocal_chain_22_cost_fn(network_ansatz, parallel=False, **qnode_kwargs):
    """For the provided ``network_ansatz``, constructs the cost function for the
    :math:`n`-local chain Bell inequality for binary inputs and outputs.

    :param network_ansatz: The ansatz for the networks. 
    :type network_ansatz: qnet.NetworkAnsatz

    :param parallel: If ``True``, remote qnode executions are made in parallel web requests.
    :type parallel: *optional* bool

    .. math::

       Cost(\\vec{\\theta}) = - \\left(\\sqrt{|I_{22}^n|} + \\sqrt{|J_{22}^n|} \\right)/2

    where the quantities :math:`I_{22}^n` and :math:`J_{22}^n` are defined as

    .. math::

       I_{22}^n &:= \\frac{1}{4} \\sum_{y_1,y_{n+1}=0}^{1}\\langle B^1_{y_1}B^2_0\\dots
           B^{n}_0B^{n+1}_{y_{n+1}}\\rangle_{22}, \\\\
       J_{22}^n &:= \\frac{1}{4} \\sum_{y_1,y_{n+1}=0}^{1}(-1)^{y_1+y_{n+1}}\\langle
           B_{y_1}^{B_{1}^2}\\dots B_{1}^{n}B_{y_{n+1}}^{n+1} \\rangle_{22},x

    and the :math:`n`-local correlator is defined as

    .. math::

       \\langle B^1_{y_1}\\dots B^{n+1}_{y_{n+1}} \\rangle_{22} = \\text{Tr}\\left[\\left(
       \\bigotimes_{j=1}^{n+1} B^j_{y_j}\\right) \\bigotimes_{i=1}^n \\rho^{A^iA^{i+1}} \\right],
    
    with :math:`B^j_{y_j}` being a dichotomic parity observable at the :math:`j^{th}` measurement node.
    Note that the :math:`n`-local correlator is simply a parity measurement distributed across
    all measurement nodes in the network.

    The maximal score for the dichotomic :math:`n` -local Bell inequality is known to be
    :math:`\\sqrt{2} \\approx 1.414 213`.

    :returns: A cost function that can be evaluated as ``cost(*network_settings)`` where
              ``network_settings`` have the appropriate dimensions for the provided ``network_ansatz``
    :rtype: Function
    """

    I22 = chain_I22_fn(network_ansatz, parallel=parallel, **qnode_kwargs)
    J22 = chain_J22_fn(network_ansatz, parallel=parallel, **qnode_kwargs)

    def cost(*network_settings):
        I22_score = I22(*network_settings)
        J22_score = J22(*network_settings)

        return -(math.sqrt(math.abs(I22_score) / 4) + math.sqrt(math.abs(J22_score) / 4))

    return cost


def parallel_nlocal_chain_grad_fn(network_ansatz, natural_grad=False, **qnode_kwargs):
    """Constructs a parallelizeable gradient function ``grad_fn`` for the :math:`n`-local
    chain cost.

    The parallelization is achieved through multithreading and intended to improve the
    efficiency of remote qnode execution.
    The number of threads is restricted to four in order to be compatible with IBM's API.

    :param network_ansatz: The ansatz describing the :math:`n`-local chain network.
    :type network_ansatz: NetworkAnsatz

    :param natural_grad: If ``True``, the natural gradient is evaluated by scaling the
                         gradient by the inverse of the metric tensor.
    :type natural_grad: *optional* Bool

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``nlocal_chain_grad(*network_settings)``.
    :rtype: function

    .. warning::
        Parallel gradient computation is flaky on PennyLane v0.28+. Intermittent  failures may occur.

    """

    from ..lazy_dask_import import dask

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    n = len(network_ansatz.layers[0])

    chain_qnodes = [global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(4)]

    I22 = chain_I22_fn(network_ansatz, parallel=True, **qnode_kwargs)
    J22 = chain_J22_fn(network_ansatz, parallel=True, **qnode_kwargs)

    I22_xy_meas_inputs = [[x] + [0 for i in range(n - 1)] + [y] for x, y in xy_vals]
    J22_xy_meas_inputs = [[x] + [1 for i in range(n - 1)] + [y] for x, y in xy_vals]

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]

    def _grad_fn(settings, qnode):
        return qml.grad(qnode)(settings)

    def _nat_grad_fn(settings, qnode):
        ginv = pinvh(qml.metric_tensor(qnode, approx="block-diag")(settings))
        return ginv @ _grad_fn(settings, qnode)

    grad_fn = _nat_grad_fn if natural_grad else _grad_fn

    def nlocal_chain_grad_fn(*network_settings):
        I22_score = I22(*network_settings)
        J22_score = J22(*network_settings)

        I22_xy_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in I22_xy_meas_inputs
        ]
        J22_xy_settings = [
            network_ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            for meas_inputs in J22_xy_meas_inputs
        ]

        I22_delayed_grads = [
            dask.delayed(grad_fn)(settings, chain_qnodes[i])
            for i, settings in enumerate(I22_xy_settings)
        ]
        J22_delayed_grads = [
            dask.delayed(grad_fn)(settings, chain_qnodes[i])
            for i, settings in enumerate(J22_xy_settings)
        ]

        I22_grads = dask.compute(*I22_delayed_grads, scheduler="threads")
        J22_grads = dask.compute(*J22_delayed_grads, scheduler="threads")

        settings_grad = network_ansatz.zero_network_settings()

        I22_scalar = -(1 / 4) * math.sign(I22_score) / math.sqrt(math.abs(I22_score))
        J22_scalar = -(1 / 4) * math.sign(J22_score) / math.sqrt(math.abs(J22_score))

        for i in range(4):
            x = I22_xy_meas_inputs[i][0]
            y = I22_xy_meas_inputs[i][-1]
            J22_sign = (-1) ** (x + y)

            settings_grad += I22_scalar * network_ansatz.expand_qnode_settings(
                I22_grads[i], [[0] * n, I22_xy_meas_inputs[i]]
            )
            settings_grad += (
                J22_sign
                * J22_scalar
                * network_ansatz.expand_qnode_settings(
                    J22_grads[i], [[0] * n, J22_xy_meas_inputs[i]]
                )
            )

        return settings_grad

    return nlocal_chain_grad_fn
