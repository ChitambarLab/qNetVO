import pennylane as qml
from pennylane import math
from ..qnodes import global_parity_expval_qnode
from scipy.linalg import pinvh


def chsh_inequality_cost_fn(network_ansatz, parallel=False, **qnode_kwargs):
    """Constructs a cost function for maximizing the score against the CHSH Bell inequality.
    This inequality is defined as

    .. math::

        I_{CHSH} = \\sum_{x,y\\in\\{0,1\\}}(-1)^{x\\cdot y}\\langle A_x B_y \\rangle

    where :math:`\\langle A_x B_y \\rangle = \\sum_{a,b\\in\\{-1,1\\}} a\\cdot b P(a,b|x,y)` is
    the two-body correlator between dichotomic observables :math:`A_x` and :math:`B_y`.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :param parallel: If ``True``, remote qnode executions are made in parallel web requests.
    :type parallel: *optional* bool

    :param qnode_kwargs: Keyword arguments used only for ``pennylane.qnode`` construction.
    :type qnode_kwargs: *optional* dict

    :returns: A cost function evaluated as ``cost(*network_settings)`` where
              the ``network_settings`` are obtained from the provided
              ``network_ansatz`` class.
    """

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]
    network_inputs = [static_prep_inputs + [xy] for xy in [[0, 0], [0, 1], [1, 0], [1, 1]]]

    if parallel:
        from ..lazy_dask_import import dask

        chsh_qnodes = [global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(4)]
    else:
        chsh_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    def chsh_cost(*network_settings):
        xy_settings = [
            network_ansatz.qnode_settings(network_settings, network_input)
            for network_input in network_inputs
        ]

        if parallel:
            delayed_results = [
                dask.delayed(chsh_qnodes[i])(settings) for i, settings in enumerate(xy_settings)
            ]

            results = math.stack(dask.compute(*delayed_results, scheduler="threads"))
        else:
            results = math.stack([chsh_qnode(settings) for settings in xy_settings])

        return -(math.sum(results * math.stack([1, 1, 1, -1])))

    return chsh_cost


def parallel_chsh_grad_fn(network_ansatz, natural_grad=False, **qnode_kwargs):
    """Constructs a parallelizeable gradient function ``grad_fn`` for the CHSH cost.

    The parallelization is achieved through multithreading and intended to improve the
    efficiency of remote qnode execution.

    The natural gradient

    .. math::

        \\nabla_{ng}I_{CHSH}(\\vec{\\theta}):= g^{-1}(\\vec{\\theta})\\nabla I_{CHSH}(\\vec{\\theta})
    
    scales the euclidean gradient :math:`\\nabla` by the pseudo-inverse of the Fubini-Study metric tensor
    :math:`g^{-1}(\\vec{\\theta})`.
    The natural gradient of the :math:`-I_{CHSH}(\\vec{\\theta})` cost function is evaluated directly as

    .. math::

        -\\nabla_{ng}I_{CHSH}(\\vec{\\theta})  &= -\\nabla_{ng}\\sum_{x,y=0}^1 (-1)^{x \\wedge y}\\langle A_x B_y\\rangle(\\vec{\\theta}_{x,y}) \\\\
        &= -\\sum_{x,y=0}^1 (-1)^{x \\wedge y}\\nabla_{ng}\\langle A_x B_y \\rangle(\\vec{\\theta_{x,y}}) \\\\
        &= -\\sum_{x,y=0}^1(-1)^{x \\wedge y}g^{-1}(\\vec{\\theta}_{x,y})\\nabla\\langle A_x B_y \\rangle(\\vec{\\theta}_{x,y})

    where :math:`\\langle A_x B_y \\rangle(\\vec{\\theta}_{x,y})` is the expectation value of observables
    :math:`A_x` and :math:`B_y` parameterized by the settings :math:`\\vec{\\theta}_{x,y}`.

    :param network_ansatz: The ansatz describing the network for which the CHSH inequality considered.
    :type network_ansatz: NetworkAnsatz

    :param natural_grad: If ``True``, then the natural gradient is evaluated. Default ``False``.
    :type natural_grad: *optional* Bool

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``grad_fn(*network_settings)``.
    :rtype: function

    .. warning::
        Parallel gradient computation is flaky on PennyLane v0.28+. Intermittent  failures may occur.
    
    """

    from ..lazy_dask_import import dask

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]
    network_inputs = [static_prep_inputs + [xy] for xy in [[0, 0], [0, 1], [1, 0], [1, 1]]]

    qnodes = []
    for _ in network_inputs:
        chsh_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)
        qnodes.append(chsh_qnode)

    def _grad_fn(settings, qnode):
        return qml.grad(qnode)(settings)

    def _nat_grad_fn(settings, qnode):
        ginv = pinvh(qml.metric_tensor(qnode, approx="block-diag")(settings))
        return ginv @ _grad_fn(settings, qnode)

    grad_fn = _nat_grad_fn if natural_grad else _grad_fn

    def parallel_chsh_grad(*network_settings):
        xy_settings = [
            network_ansatz.qnode_settings(network_settings, input) for input in network_inputs
        ]

        delayed_grads = [
            dask.delayed(grad_fn)(settings, qnodes[i]) for i, settings in enumerate(xy_settings)
        ]
        grads = dask.compute(*delayed_grads, scheduler="threads")

        grad = network_ansatz.zero_network_settings()
        for grad_id, inputs in enumerate(network_inputs):
            x = inputs[1][0]
            y = inputs[1][1]

            scalar = -1 * (-1) ** (x * y)
            grad += scalar * network_ansatz.expand_qnode_settings(grads[grad_id], inputs)

        return grad

    return parallel_chsh_grad
