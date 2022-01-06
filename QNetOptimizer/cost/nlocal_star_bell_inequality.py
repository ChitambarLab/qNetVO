import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from .qnodes import global_parity_expval_qnode


def star_I22_fn(network_ansatz, **qnode_kwargs):
    """Constructs a network-specific ``I22(scenario_settings)`` function that
    evaluates the :math:`I^n_{22}` quantity for the :math:`n`-local star network.

    The :math:`I^n_{22}` quantity is formally expressed as

    .. math::

        I^n_{22} = \\frac{1}{2^n}\\sum_{x_1,\\dots,x_n}\\langle A_{x_1}\\dots A_{x_n}B_0\\rangle,

    where :math:`x_i\\in\\{0,1\\}` and :math:`A_{x_i}` and :math:`B_{x_{n+1}}`
    are dichotomic observables.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: QNopt.NetworkAnsatz

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type: *optional* dictionary

    :returns: A function callable as ``I22(scenario_settings)`` that evaluates the :math:`I^n_{22}` quantity.
    :rtype: function
    """
    n = len(network_ansatz.prepare_nodes)

    prep_inputs = [0] * n
    input_x_vals = [[int(bit) for bit in np.binary_repr(x, width=n) + "0"] for x in range(2 ** (n))]

    star_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    def I22(scenario_settings):

        I22_x_settings = [
            network_ansatz.qnode_settings(scenario_settings, prep_inputs, meas_inputs)
            for meas_inputs in input_x_vals
        ]

        I22_results = math.array([star_qnode(settings) for settings in I22_x_settings])

        return math.sum(I22_results) / 2 ** n

    return I22


def star_J22_fn(network_ansatz, **qnode_kwargs):
    """Constructs a network-specific ``J22(scenario_settings)`` function that
    evaluates the :math:`J^n_{22}` quantity for the :math:`n`-local star network.

    The :math:`J^n_{22}` quantity is formally expressed as

    .. math::

        J^n_{22} = \\frac{1}{2^n}\\sum_{x_1,\\dots,x_n}(-1)^{\\sum_i x_i}\\langle A_{x_1}\\dots A _{x_n}B_1\\rangle,

    where :math:`x_i\\in\\{0,1\\}` and :math:`A_{x_i}` and :math:`B_{x_{n+1}}` are
    dichotomic observables.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: QNopt.NetworkAnsatz

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type: *optional* dictionary

    :returns: A function callable as ``J22(scenario_settings)`` that evaluates the :math:`J^n_{22}`
              quantity for the given ``scenario_settings``.
    :rtype: function
    """

    n = len(network_ansatz.prepare_nodes)

    prep_inputs = [0] * n
    input_x_vals = [[int(bit) for bit in np.binary_repr(x, width=n) + "1"] for x in range(2 ** (n))]

    star_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    def J22(scenario_settings):

        J22_x_settings = [
            network_ansatz.qnode_settings(scenario_settings, prep_inputs, meas_inputs)
            for meas_inputs in input_x_vals
        ]

        J22_expvals = math.array([star_qnode(settings) for settings in J22_x_settings])

        J22_scalars = math.array(
            [(-1) ** (math.sum(input_vals[0:n])) for input_vals in input_x_vals]
        )

        return math.sum(J22_scalars * J22_expvals) / 2 ** n

    return J22


def nlocal_star_22_cost_fn(network_ansatz, **qnode_kwargs):
    """A network-specific constructor for the :math:`n`-local star Bell
    inequality for scenarios when all measurement devices in the star network
    have 2 inputs and 2 outputs.

    The :math:`n`-local star network Bell inequality is expressed as

    .. math::

        |I^n_{22}|^{1/n} + |J^n_{22}|^{1/n} \\leq 1

    where the quantities :math:`I^n_{22}` and :math:`J^n_{22}` are evaluated using
    functions constructed by the :meth:`QNetOptimizer.star_I22_fn` and
    :meth:`QNetOptimizer.star_J22_fn` methods respectively.
    The classical bound is found to be 1, but quantum systems can score as high as
    :math:`\\sqrt{2}`.

    :param network_ansatz: The :math:`n`-local star network ansatz.
    :type network_ansatz: QNopt.NetworkAnsatz

    :param qnode_kwargs: keyword args passed through to the QNode constructor.
    :type: *optional* dictionary

    :returns: A function callable as ``nlocal_star_22_cost(scenario_settings)`` that evaluates
              the cost as :math:`-|I^n_{22}|^{1/n} - |J^n_{22}|^{1/n}`.
    :rtype: function
    """

    n = len(network_ansatz.prepare_nodes)

    I22 = star_I22_fn(network_ansatz, **qnode_kwargs)
    J22 = star_J22_fn(network_ansatz, **qnode_kwargs)

    def cost(scenario_settings):

        I22_score = I22(scenario_settings)
        J22_score = J22(scenario_settings)

        return -(np.power(math.abs(I22_score), 1 / n) + np.power(math.abs(J22_score), 1 / n))

    return cost
