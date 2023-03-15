from pennylane import math
from pennylane import numpy as np
from ..qnodes import joint_probs_qnode, global_parity_expval_qnode
from ..utilities import mixed_base_num, ragged_reshape


def linear_probs_cost_fn(network_ansatz, game, postmap=np.array([]), qnode_kwargs={}):
    """Constructs an ansatz-specific cost that is a linear function of the network probablities.

    The cost function is encoded into a ``game`` matrix whose
    coefficients scale conditional probabilities for the network.
    The cost is derived from the score which is evaluated as,

    .. math::

        \\langle\\mathbf{G},\\mathbf{P}\\rangle = \\sum_{\\{\\vec{x}_i\\}_i,\\vec{a}} G_{a|\\{x_i\\}_i} P(a|\\{x_i\\}),

    where :math:`\\mathbf{P}` is a behavior (see :meth:`qnetvo.behavior_fn`) and :math:`\\{\\vec{x}_i\\}_i` specifies
    the collection of inputs for each layer as indexed by :math:`i`.

    A post-processing map :math:`\\mathbf{L}` may optionally be applied as
    :math:`\\mathbf{L}\\mathbf{P}_{Net}` where

    .. math::

        \\mathbf{L} = \\sum_{z',z}P(z'|z)|z'\\rangle\\langle z|.

    In the above expression, :math:`z'` is a new output drawn from a new alphabet.

    :param network_ansatz: The network to which the cost function is applied.
    :type network_ansatz: ``NetworkAnsatz`` class

    :param game: A matrix with dimensions ``A x (\\prod_i X_i)`` for
    :type game: np.arrray

    :param postmap: A post-processing map applied to the bitstrings output from the
                     quantum circuit. The ``postmap`` matrix is column stochastic, that is,
                     each column sums to one and contains only positive values.
    :type postmap: *optional* np.ndarray

    :returns: A cost function evaluated as ``cost(*network_settings)``.
    :rtype: function

    :raises ValueError: If the number of outputs from the qnode do not match the
        the number of rows in the specified ``game``.
    """

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    net_num_in = math.prod(network_ansatz.layers_total_num_in)
    num_inputs_list = math.concatenate(network_ansatz.layers_node_num_in).tolist()
    node_input_ids = [
        ragged_reshape(mixed_base_num(i, num_inputs_list), network_ansatz.layers_num_nodes)
        for i in range(net_num_in)
    ]

    raw_net_num_out = 2 ** len(network_ansatz.layers_wires[-1])

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

    def cost(*network_settings):
        score = 0
        for i, input_id_set in enumerate(node_input_ids):
            settings = network_ansatz.qnode_settings(network_settings, input_id_set)

            raw_probs = probs_qnode(settings)
            probs = postmap @ raw_probs if has_postmap else raw_probs

            score += math.sum(game[:, i] * probs)

        return -(score)

    return cost
