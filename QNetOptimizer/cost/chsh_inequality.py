import dask
import pennylane as qml
from pennylane import math
from .qnodes import global_parity_expval_qnode
from scipy.linalg import pinvh


def chsh_inequality_cost(chsh_ansatz, parallel=False, qnode_kwargs={}):
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
        chsh_qnodes = []
        for xy in xy_vals:
            chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)
            chsh_qnodes.append(chsh_qnode)
    else:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)

    def chsh_cost(scenario_settings):

        prep_settings = chsh_ansatz.layer_settings(scenario_settings[0], [0])
        xy_meas_settings = [chsh_ansatz.layer_settings(scenario_settings[1], xy) for xy in xy_vals]

        if parallel:
            delayed_results = [
                dask.delayed(chsh_qnodes[i])(prep_settings, meas_settings)
                for i, meas_settings in enumerate(xy_meas_settings)
            ]

            results = math.array(dask.compute(*delayed_results, scheduler="threads"))
        else:
            results = math.array(
                [chsh_qnode(prep_settings, meas_settings) for meas_settings in xy_meas_settings]
            )

        return -(math.sum(results * math.array([1, 1, 1, -1])))

    return chsh_cost


def parallel_chsh_grad(chsh_ansatz, qnode_kwargs={}):
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

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    num_settings_A = chsh_ansatz.measure_nodes[0].num_settings

    qnode_grads = []
    for xy in xy_vals:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)
        qnode_grads.append(qml.grad(chsh_qnode))

    def grad_fn(scenario_settings):

        prep_settings = chsh_ansatz.layer_settings(scenario_settings[0], [0])
        xy_meas_settings = [chsh_ansatz.layer_settings(scenario_settings[1], xy) for xy in xy_vals]

        delayed_results = [
            dask.delayed(qnode_grads[i])(prep_settings, meas_settings)
            for i, meas_settings in enumerate(xy_meas_settings)
        ]

        results = dask.compute(*delayed_results, scheduler="threads")

        grad = chsh_ansatz.zero_scenario_settings()

        results_id = 0
        for x, y in xy_vals:
            scalar = -1 * (-1) ** (x * y)

            result = results[results_id]
            grad[0][0][0] += scalar * result[0]
            grad[1][0][x] += scalar * result[1][0:num_settings_A]
            grad[1][1][y] += scalar * result[1][num_settings_A:]

            results_id += 1

        return grad

    return grad_fn


def chsh_natural_grad(chsh_ansatz, qnode_kwargs={}):
    """Constructs a parallelized natural gradient function ``natural_grad`` for the CHSH
    cost function.
    The parallelization is achieved through multithreading and intended to improve the
    efficiency of remote qnode execution.

    :param chsh_ansatz: The ansatz describing the network for which the CHSH inequality considered.
    :type chsh_ansatz: NetworkAnsatz

    :param qnode_kwargs: A keyword argument passthrough to qnode construction.
    :type qnode_kwargs: *optional* dict

    :returns: A parallelized (multithreaded) gradient function ``grad_fn(scenario_settings)``.
    :rtype: function
    """

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    num_settings_A = chsh_ansatz.measure_nodes[0].num_settings
    num_prep_settings = chsh_ansatz.prepare_nodes[0].num_settings

    qnode_qgrads = []
    for xy in xy_vals:
        chsh_qnode = global_parity_expval_qnode(chsh_ansatz, **qnode_kwargs)
        qnode_qgrads.append((chsh_qnode, qml.grad(chsh_qnode)))

    def natural_grad(scenario_settings):
        prep_settings = chsh_ansatz.layer_settings(scenario_settings[0], [0])
        xy_meas_settings = [chsh_ansatz.layer_settings(scenario_settings[1], xy) for xy in xy_vals]

        def _ng(prep_settings, meas_settings, qnode, qgrad):
            grad = math.append(*qgrad(prep_settings, meas_settings))
            ginv = pinvh(chsh_qnode.metric_tensor(prep_settings, meas_settings))

            return ginv @ grad

        delayed_results = [
            dask.delayed(_ng)(prep_settings, meas_settings, qnode_qgrads[i][0], qnode_qgrads[i][1])
            for i, meas_settings in enumerate(xy_meas_settings)
        ]

        nat_grads = dask.compute(*delayed_results, scheduler="threads")

        grad = chsh_ansatz.zero_scenario_settings()
        qnode_id = 0
        for x, y in xy_vals:

            (qnode, qgrad) = qnode_qgrads[qnode_id]

            meas_settings = chsh_ansatz.layer_settings(scenario_settings[1], [x, y])

            scalar = -1 * (-1) ** (x * y)

            # nat_grad = _ng(prep_settings, meas_settings, qnode, qgrad)
            nat_grad = nat_grads[qnode_id]

            # remapping states and measurements
            grad[0][0][0] += scalar * nat_grad[0:num_prep_settings]
            grad[1][0][x] += (
                scalar * nat_grad[num_prep_settings : num_prep_settings + num_settings_A]
            )
            grad[1][1][y] += scalar * nat_grad[num_prep_settings + num_settings_A :]

            qnode_id += 1

        return grad

    return natural_grad
