from pennylane import math
from .qnodes import global_parity_expval_qnode


def nlocal_chain_cost_22(network_ansatz, **qnode_kwargs):
    """For the provided ``network_ansatz``, constructs the cost function for the
    :math:`n`-local chain Bell inequality for binary inputs and outputs.

    :param network_ansatz: The ansatz for the networks. 
    :type network_ansatz: QNopt.NetworkAnsatz

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

    nlocal_chain_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    num_interior_nodes = len(network_ansatz.measure_nodes) - 2

    def cost(scenario_settings):
        prep_settings, meas_settings = scenario_settings

        static_prep_settings = network_ansatz.layer_settings(
            prep_settings, [0] * len(network_ansatz.prepare_nodes)
        )

        I22_score = 0
        J22_score = 0

        for x_a, x_b in [(0, 0), (0, 1), (1, 0), (1, 1)]:

            I22_inputs = [x_a] + [0 for i in range(num_interior_nodes)] + [x_b]
            # I22_meas_settings = network_ansatz.layer_settings(meas_settings, I22_inputs)
            # I22_score += nlocal_chain_qnode(static_prep_settings, I22_meas_settings)
            I22_settings = network_ansatz.qnode_settings(scenario_settings, [0], I22_inputs)
            I22_score += nlocal_chain_qnode(I22_settings)

            # J22_inputs = [x_a] + [1 for i in range(num_interior_nodes)] + [x_b]
            # J22_meas_settings = network_ansatz.layer_settings(meas_settings, J22_inputs)
            J22_inputs = [x_a] + [1 for i in range(num_interior_nodes)] + [x_b]
            J22_settings = network_ansatz.qnode_settings(scenario_settings, [0], J22_inputs)
            J22_score += ((-1) ** (x_a + x_b)) * nlocal_chain_qnode(J22_settings)

        return -(math.sqrt(math.abs(I22_score) / 4) + math.sqrt(math.abs(J22_score) / 4))

    return cost
