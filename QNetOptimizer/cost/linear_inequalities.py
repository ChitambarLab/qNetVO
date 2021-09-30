from pennylane import math
from .qnodes import joint_probs_qnode


def linear_probs_cost(network_ansatz, game, **qnode_kwargs):
    """Constructs a generic linear cost function.
        The linear cost function is encoded into a ``game`` matrix whose
        coefficients scale conditional probabilities for the network.

        .. math::

            \\sum_{\\vec{x},\\vec{y},\\vec{b}} G_{b|x,y} P(b|x,y)

        :param network_ansatz: The network to which the cost function is applied.
    :type network_ansatz: ``NetworkAnsatz`` class

    :param game: A matrix with dimensions ``B x (X x Y)`` for
    :type game: np.arrray

    :returns: A cost function evaluated as ``cost(prepare_settings, measure_settings)``.
    :rtype: function
    """

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    # find non-zero columns of game matrix
    num_in_prep_nodes = [node.num_in for node in network_ansatz.prepare_nodes]
    num_in_meas_nodes = [node.num_in for node in network_ansatz.measure_nodes]

    num_inputs = math.prod(num_in_prep_nodes) * math.prod(num_in_meas_nodes)

    game_inputs = game.shape[1]

    if game_inputs != num_inputs:
        raise ValueError("game matrix rows must have dimension " + num_inputs + ".")

    # convert coefficient ids into a list of prep/meas node inputs
    base_digits = num_in_prep_nodes + num_in_meas_nodes
    node_input_ids = [mixed_base_convert(i, base_digits) for i in range(num_inputs)]

    def cost(scenario_settings):
        score = 0
        for (i, input_id_set) in enumerate(node_input_ids):

            # construct layer settings for each non-zero coefficient
            prep_settings = network_ansatz.layer_settings(
                scenario_settings[0], input_id_set[0 : len(network_ansatz.prepare_nodes)]
            )
            meas_settings = network_ansatz.layer_settings(
                scenario_settings[1], input_id_set[len(network_ansatz.prepare_nodes) :]
            )

            probs = probs_qnode(prep_settings, meas_settings)

            score += math.sum(game[:, i] * probs)

        return -(score)

    return cost


def mixed_base_convert(n, base_digits):
    """Converts a base-10 number ``n`` into a mixed base number with digit
    values described by the ``base_digits`` array.

    :param n: A base-10 number
    :type n: int

    :param base_digits: A list of integers representing the largest value for each
                        digit in the mixed base number
    :type base_digits: list[int]

    :returns: A list of integers representing the mixed base number.
    :rtype: list[int]

    """
    mixed_base_values = []
    n_tmp = n
    for i in range(len(base_digits)):
        place = int(math.prod(base_digits[i + 1 :]))

        mixed_base_values += [n_tmp // place]
        n_tmp = n_tmp % place

    return mixed_base_values
