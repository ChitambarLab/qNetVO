import dask
from pennylane import math
from .qnodes import global_parity_expval_qnode


def nlocal_chain_cost_22(network_ansatz, parallel=False, **qnode_kwargs):
    """For the provided ``network_ansatz``, constructs the cost function for the
    :math:`n`-local chain Bell inequality for binary inputs and outputs.

    :param network_ansatz: The ansatz for the networks. 
    :type network_ansatz: QNopt.NetworkAnsatz

    :param parallel: If ``True``, remote qnode executions are made in parallel web requests.
    :type parallel: *optional* bool

    .. math::

       Cost(\\vec{\\theta}) = - \\left(\\sqrt{|I_{22}^n|} + \\sqrt{|J_{22}^n|} \\right)

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

    :returns: A cost function that can be evaluated as ``cost(scenario_settings)`` where
              ``scenario_settings`` have the appropriate dimensions for the provided ``network_ansatz``
    :rtype: Function
    """

    xy_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]

    if parallel:
        chain_qnodes = [global_parity_expval_qnode(network_ansatz, **qnode_kwargs) for i in range(4)]
    else:
        chain_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    num_interior_nodes = len(network_ansatz.measure_nodes) - 2
    prep_inputs = [0] * len(network_ansatz.prepare_nodes)

    def cost(scenario_settings):

        I22_score = 0
        J22_score = 0

        I22_xy_inputs = [[x_a] + [0 for i in range(num_interior_nodes)] + [x_b] for x_a, x_b in xy_vals]
        J22_xy_inputs = [[x_a] + [1 for i in range(num_interior_nodes)] + [x_b] for x_a, x_b in xy_vals]

        I22_xy_settings = [network_ansatz.qnode_settings(scenario_settings, prep_inputs, meas_inputs) for meas_inputs in I22_xy_inputs]
        J22_xy_settings = [network_ansatz.qnode_settings(scenario_settings, prep_inputs, meas_inputs) for meas_inputs in J22_xy_inputs]

        if parallel:
            I22_delayed_results = [
                dask.delayed(chain_qnodes[i])(settings) for i, settings in enumerate(I22_xy_settings)
            ]
            J22_delayed_results = [
                dask.delayed(chain_qnodes[i])(settings) for i, settings in enumerate(J22_xy_settings)
            ]

            # IBM only allows 5 parallel requests (we do two batches of 4 and 4).
            I22_results = math.array(dask.compute(*I22_delayed_results, scheduler="threads"))
            J22_results = math.array(dask.compute(*J22_delayed_results, scheduler="threads"))
        else:
            I22_results = math.array([chain_qnode(settings) for settings in I22_xy_settings])
            J22_results = math.array([chain_qnode(settings) for settings in J22_xy_settings])

        I22_score = math.sum(I22_results)
        J22_score = math.sum(math.array([1,-1,-1,1]) * J22_results)

        return -(math.sqrt(math.abs(I22_score) / 4) + math.sqrt(math.abs(J22_score) / 4))

    return cost
