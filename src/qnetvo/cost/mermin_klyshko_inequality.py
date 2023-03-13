import pennylane as qml
from pennylane import math
from ..qnodes import global_parity_expval_qnode


def mermin_klyshko_inputs_scalars(n):
    """Helper function for handling the algebra of combining correlator terms
    in the Mermin-Klyshko (MK) inequality.

    This function supports :meth:`mermin_klyshko_cost_fn`.

    :param n: The number of measurement nodes in the scenario.
    :type n: Int

    :returns: The first element of the returned tuple is a list of
              measurement inputs :math:`y_j\\in\\{0,1\\}` for the MK
              correlator terms. The second element of the returned tuple
              is a list of scalar multipliers for each correlator term.
    :rtype: Tuple(List[Int], List[Int])
    """

    inputs_list = [[0]]
    scalars_list = [1]

    for i in range(1, n):
        inputs_list_tmp = []
        scalars_list_tmp = []

        append_inputs = [0, 1, 0, 1]
        append_scalars = [1, 1, 1, -1]

        for i in range(len(inputs_list)):
            inputs = inputs_list[i]
            not_inputs = [(x + 1) % 2 for x in inputs]

            scalar = scalars_list[i]

            for i, append_input in enumerate(append_inputs):
                new_scalar_term = scalar * append_scalars[i]
                new_inputs = (inputs if i < 2 else not_inputs) + [append_input]

                if new_inputs in inputs_list_tmp:
                    inputs_id = inputs_list_tmp.index(new_inputs)

                    new_scalar = new_scalar_term + scalars_list_tmp[inputs_id]
                    if new_scalar == 0:
                        scalars_list_tmp.pop(inputs_id)
                        inputs_list_tmp.pop(inputs_id)
                    else:
                        scalars_list_tmp[inputs_id] = new_scalar
                else:
                    inputs_list_tmp.append(new_inputs)
                    scalars_list_tmp.append(new_scalar_term)

        inputs_list = inputs_list_tmp
        scalars_list = scalars_list_tmp

    return inputs_list, scalars_list


def mermin_klyshko_cost_fn(ansatz, **qnode_kwargs):
    """Constructs an ansatz-specific cost function based upon the
    Mermin-Klyshko (MK) inequality.

    :param ansatz: The network ansatz for which to apply the MK inequality.
    :type ansatz: NetworkAnsatz

    :param qnode_kwargs: Keyword arguments passed through to the qnode constructors.

    :returns: A cost function, ``cost(*network_settings)``, that evaluates :math:`-I_{\\text{MK}}`
              for the supplied network settings.
    :rtype: Function
    """
    mk_qnode = global_parity_expval_qnode(ansatz, **qnode_kwargs)

    num_meas_nodes = len(ansatz.layers[-1])
    meas_inputs_list, scalars_list = mermin_klyshko_inputs_scalars(num_meas_nodes)

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in ansatz.layers[0:-1]]

    def cost(*network_settings):
        score = 0

        num_correlators = len(meas_inputs_list)

        for i in range(num_correlators):
            meas_inputs = meas_inputs_list[i]
            scalar = scalars_list[i]

            settings = ansatz.qnode_settings(network_settings, static_prep_inputs + [meas_inputs])
            score += scalar * mk_qnode(settings)

        return -(score)

    return cost


def mermin_klyshko_classical_bound(n):
    """The classical bound for the Mermin-Klyshko inequality is :math:`2^{n-1}`.

    :param n: The number of measurement nodes.
    :type n: Int

    :returns: The classical bound.
    :rtype: Float
    """
    return 2 ** (n - 1)


def mermin_klyshko_quantum_bound(n):
    """The quantum bound for the Mermin-Klyshko inequality is :math:`2^{3(n-1)/2}`.

    :param n: The number of measurement nodes.
    :type n: Int

    :returns: The quantum bound.
    :rtype: Float
    """
    return 2 ** (3 * (n - 1) / 2)
