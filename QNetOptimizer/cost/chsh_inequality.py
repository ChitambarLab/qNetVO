import dask
from pennylane import math
from .qnodes import global_parity_expval_qnode


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
