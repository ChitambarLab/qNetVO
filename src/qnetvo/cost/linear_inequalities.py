from pennylane import math
from pennylane import numpy as np
from ..qnodes import joint_probs_qnode, global_parity_expval_qnode
from ..utilities import mixed_base_num


def linear_probs_cost_fn(network_ansatz, game, postmap=np.array([]), qnode_kwargs={}):
    """Constructs an ansatz-specific cost that is a linear function of the network probablities.

    The cost function is encoded into a ``game`` matrix whose
    coefficients scale conditional probabilities for the network.
    The cost is derived from the score which is evaluated as,

    .. math::

        \\langle\\mathbf{G},\\mathbf{P}\\rangle = \\sum_{\\vec{x},\\vec{y},\\vec{b}} G_{b|x,y} P(b|x,y),

    where :math:`\\mathbf{P}` is a behavior (see :meth:`qnetvo.behavior_fn`).

    A post-processing map :math:`\\mathbf{L}` may optionally be applied as
    :math:`\\mathbf{L}\\mathbf{P}_{Net}` where

    .. math::

        \\mathbf{L} = \\sum_{z',z}P(z'|z)|z'\\rangle\\langle z|.

    In the above expression, :math:`z'` is a new output drawn from a new alphabet.

    :param network_ansatz: The network to which the cost function is applied.
    :type network_ansatz: ``NetworkAnsatz`` class

    :param game: A matrix with dimensions ``B x (X x Y)`` for
    :type game: np.arrray

    :param postmap: A post-processing map applied to the bitstrings output from the
                     quantum circuit. The ``postmap`` matrix is column stochastic, that is,
                     each column sums to one and contains only positive values.
    :type postmap: *optional* np.ndarray

    :returns: A cost function evaluated as ``cost(prepare_settings, measure_settings)``.
    :rtype: function

    :raises ValueError: If the number of outputs from the qnode do not match the
        the number of rows in the specified ``game``.
    """

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    num_in_prep_nodes = [node.num_in for node in network_ansatz.prepare_nodes]
    num_in_meas_nodes = [node.num_in for node in network_ansatz.measure_nodes]
    num_out_meas_nodes = [node.num_out for node in network_ansatz.measure_nodes]

    net_num_in = math.prod(num_in_prep_nodes) * math.prod(num_in_meas_nodes)
    net_num_out = math.prod(num_out_meas_nodes)

    raw_net_num_out = 2 ** len(network_ansatz.measure_wires)

    game_outputs, game_inputs = game.shape

    if game_inputs != net_num_in:
        raise ValueError("The `game` matrix must have " + str(net_num_in) + " columns.")

    has_postmap = len(postmap) != 0
    if not (has_postmap):
        if game_outputs != raw_net_num_out:
            raise ValueError(
                "The `game` matrix must either have "
                + str(raw_net_num_out)
                + " rows, or a `postmap` is needed."
            )
    else:
        if postmap.shape[0] != game_outputs:
            raise ValueError("The `postmap` must have " + str(game_outputs) + " rows.")
        elif postmap.shape[1] != raw_net_num_out:
            raise ValueError("The `postmap` must have " + str(raw_net_num_out) + " columns.")

    # convert coefficient ids into a list of prep/meas node inputs
    base_digits = num_in_prep_nodes + num_in_meas_nodes
    node_input_ids = [mixed_base_num(i, base_digits) for i in range(net_num_in)]

    def cost(scenario_settings):
        score = 0
        for (i, input_id_set) in enumerate(node_input_ids):
            settings = network_ansatz.qnode_settings(
                scenario_settings,
                input_id_set[0 : len(network_ansatz.prepare_nodes)],
                input_id_set[len(network_ansatz.prepare_nodes) :],
            )

            raw_probs = probs_qnode(settings)
            probs = postmap @ raw_probs if has_postmap else raw_probs

            score += math.sum(game[:, i] * probs)

        return -(score)

    return cost
