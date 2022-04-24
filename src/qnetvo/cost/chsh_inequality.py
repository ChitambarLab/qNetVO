import pennylane as qml
from pennylane import math
from ..qnodes import global_parity_expval_qnode
from scipy.linalg import pinvh


def chsh_inequality_cost(chsh_ansatz, parallel=False, **qnode_kwargs):
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

    :returns: A cost function evaluated as ``cost(scenario_settings)`` where
              the ``scenario_settings`` are obtained from the provided
              ``network_ansatz`` class.
    """

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]

    if parallel:
        from ..lazy_dask_import import dask

        chsh_qnodes = [global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs) for i in range(4)]
    else:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)

    def chsh_cost(scenario_settings):

        xy_settings = [chsh_ansatz.qnode_settings(scenario_settings, [0], xy) for xy in xy_vals]

        if parallel:
            delayed_results = [
                dask.delayed(chsh_qnodes[i])(settings) for i, settings in enumerate(xy_settings)
            ]

            results = math.array(dask.compute(*delayed_results, scheduler="threads"))
        else:
            results = math.array([chsh_qnode(settings) for settings in xy_settings])

        return -(math.sum(results * math.array([1, 1, 1, -1])))

    return chsh_cost


def parallel_chsh_grad(chsh_ansatz, **qnode_kwargs):
    """Constructs a parallelizeable gradient function ``grad_fn`` for the CHSH
    cost.

    The parallelization is achieved through multithreading and intended to improve the
    efficiency of remote qnode execution.

    :param chsh_ansatz: The ansatz describing the network for which the CHSH inequality considered.
    :type chsh_ansatz: NetworkAnsatz

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``grad_fn(scenario_settings)``.
    :rtype: function
    """

    from ..lazy_dask_import import dask

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    num_settings_A = chsh_ansatz.measure_nodes[0].num_settings
    num_prep_settings = chsh_ansatz.prepare_nodes[0].num_settings

    qnode_grads = []
    for xy in xy_vals:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)
        qnode_grads.append(qml.grad(chsh_qnode))

    def grad_fn(scenario_settings):

        xy_settings = [chsh_ansatz.qnode_settings(scenario_settings, [0], xy) for xy in xy_vals]

        delayed_results = [
            dask.delayed(qnode_grads[i])(settings) for i, settings in enumerate(xy_settings)
        ]

        results = dask.compute(*delayed_results, scheduler="threads")

        grad = chsh_ansatz.zero_scenario_settings()

        results_id = 0
        for x, y in xy_vals:
            scalar = -1 * (-1) ** (x * y)

            result = results[results_id]
            grad[0][0][0] += scalar * result[0:num_prep_settings]
            grad[1][0][x] += scalar * result[num_prep_settings : num_prep_settings + num_settings_A]
            grad[1][1][y] += scalar * result[num_prep_settings + num_settings_A :]

            results_id += 1

        return grad

    return grad_fn


def chsh_natural_grad(chsh_ansatz, **qnode_kwargs):
    """Constructs a parallelized natural gradient function ``natural_grad`` for the CHSH
    cost function.

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

    :param chsh_ansatz: The ansatz describing the network for which the CHSH inequality considered.
    :type chsh_ansatz: NetworkAnsatz

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``grad_fn(scenario_settings)``.
    :rtype: function
    """

    from ..lazy_dask_import import dask

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    num_settings_A = chsh_ansatz.measure_nodes[0].num_settings
    num_prep_settings = chsh_ansatz.prepare_nodes[0].num_settings

    qnodes = []
    for xy in xy_vals:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)
        qnodes.append(chsh_qnode)

    def natural_grad(scenario_settings):
        xy_settings = [chsh_ansatz.qnode_settings(scenario_settings, [0], xy) for xy in xy_vals]

        def _ng(settings, qnode):
            grad = qml.grad(qnode)(settings)
            ginv = pinvh(qml.metric_tensor(qnode, approx="block-diag")(settings))

            return ginv @ grad

        delayed_results = [
            dask.delayed(_ng)(settings, qnodes[i]) for i, settings in enumerate(xy_settings)
        ]

        nat_grads = dask.compute(*delayed_results, scheduler="threads")

        grad = chsh_ansatz.zero_scenario_settings()
        grad_id = 0
        for x, y in xy_vals:

            scalar = -1 * (-1) ** (x * y)

            nat_grad = nat_grads[grad_id]

            # remapping states and measurements
            grad[0][0][0] += scalar * nat_grad[0:num_prep_settings]
            grad[1][0][x] += (
                scalar * nat_grad[num_prep_settings : num_prep_settings + num_settings_A]
            )
            grad[1][1][y] += scalar * nat_grad[num_prep_settings + num_settings_A :]

            grad_id += 1
        return grad

    return natural_grad
