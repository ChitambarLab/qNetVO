from pennylane import math
from pennylane import numpy as np
from .qnodes import joint_probs_qnode, global_parity_expval_qnode
from ..utilities import mixed_base_num


def linear_probs_cost(network_ansatz, game, post_processing_map=np.array([]), qnode_kwargs={}):
    """Constructs a generic linear cost function on the conditional probabilities
    for the network.
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

    :raises ValueError: If the number of outputs from the qnode do not match the
        the number of rows in the specified ``game``.
    """

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)
    # parity_qnode = global_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    num_in_prep_nodes = [node.num_in for node in network_ansatz.prepare_nodes]
    num_in_meas_nodes = [node.num_in for node in network_ansatz.measure_nodes]
    num_out_meas_nodes = [node.num_out for node in network_ansatz.measure_nodes]

    net_num_in = math.prod(num_in_prep_nodes) * math.prod(num_in_meas_nodes)
    net_num_out = math.prod(num_out_meas_nodes)

    raw_net_num_out = 2 ** len(network_ansatz.measure_wires)

    game_outputs, game_inputs = game.shape

    if game_inputs != net_num_in or game_outputs != net_num_out:
        raise ValueError("`game` matrix must have dimension ("
            + str(net_num_out)
            + ", "
            + str(net_num_in)
            + ")."
        )

    if game_outputs != raw_net_num_out:
        if post_processing_map.shape[0] != net_num_out:
            raise ValueError(
                "The number of rows in the `post_processing_map` must be " + str(net_num_out) + "."
            )
        elif post_processing_map.shape[1] != raw_net_num_out:
            raise ValueError(
                "The number of columns in the `post_processing_map` must be "
                + str(raw_net_num_out)
                + "."
            )

    # convert coefficient ids into a list of prep/meas node inputs
    base_digits = num_in_prep_nodes + num_in_meas_nodes
    node_input_ids = [mixed_base_num(i, base_digits) for i in range(net_num_in)]

    def cost(scenario_settings):
        score = 0
        for (i, input_id_set) in enumerate(node_input_ids):

            prep_settings = network_ansatz.layer_settings(
                scenario_settings[0], input_id_set[0 : len(network_ansatz.prepare_nodes)]
            )
            meas_settings = network_ansatz.layer_settings(
                scenario_settings[1], input_id_set[len(network_ansatz.prepare_nodes) :]
            )

            raw_probs = probs_qnode(prep_settings, meas_settings)
            probs = raw_probs if net_num_out == raw_net_num_out else post_processing_map @ raw_probs 
   
            score += math.sum(game[:, i] * probs)

        return -(score)

    return cost
