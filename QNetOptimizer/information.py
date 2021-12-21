from pennylane import math
from .cost.qnodes import joint_probs_qnode
from .utilities import mixed_base_num
from pennylane import numpy as np


def behavior_fn(network_ansatz, postmap=np.array([]), qnode_kwargs={}):
    """Creates an ansatz-specific function for constructing the behavior matrix.

    A behavior function is created as ``P_Net = behavior(network_ansatz)``
    and called as ``P_Net(ansatz_settings)``.
    The network behavior ``P_Net`` is a column stochastic matrix containing the
    conditional probabilities,

    .. math::

        \\mathbf{P}_{Net} = \\sum_{x,y,z}P(z|x,y)|z\\rangle\\langle x,y|,


    where :math:`P(z|x,y)` is evaluated by a qnode for each set of inputs
    :math:`x` and :math:`y`.

    A post-processing map :math:`\\mathbf{L}` may optionally be applied as
    :math:`\\mathbf{L}\\mathbf{P}_{Net}` where

    .. math::

        \\mathbf{L} = \\sum_{z',z}P(z'|z)|z'\\rangle\\langle z|.


    In the above expression, :math:`z'` is a new output drawn from a new alphabet.

    :param network_ansatz: A class describing the particular quantum network.
    :type network_ansatz: NetworkAnsatz

    :param postmap: A post-processing map applied to the bitstrings output from the
                     quantum circuit. The ``postmap`` matrix is column stochastic, that is,
                     each column sums to one and contains only positive values.
    :type postmap: *optional* np.ndarray

    :returns: A function ``P_Net(scenario_settings)`` that evaluates the
              behavior matrix for a given set of settings.
    :rtype: function
    """
    num_in_prep_nodes = [node.num_in for node in network_ansatz.prepare_nodes]
    num_in_meas_nodes = [node.num_in for node in network_ansatz.measure_nodes]

    num_out_meas_nodes = [node.num_out for node in network_ansatz.measure_nodes]

    base_digits = num_in_prep_nodes + num_in_meas_nodes
    net_num_in = math.prod(base_digits)
    net_num_out = math.prod(num_out_meas_nodes)

    raw_net_num_out = 2 ** len(network_ansatz.measure_wires)

    has_postmap = len(postmap) != 0
    if has_postmap:
        if postmap.shape[1] != raw_net_num_out:
            raise ValueError("The `postmap` must have " + str(raw_net_num_out) + " columns.")

    node_input_ids = [mixed_base_num(i, base_digits) for i in range(net_num_in)]

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    def behavior(scenario_settings):
        raw_behavior = np.zeros((raw_net_num_out, net_num_in))
        for (i, input_id_set) in enumerate(node_input_ids):
            settings = network_ansatz.qnode_settings(
                scenario_settings,
                input_id_set[0 : len(num_in_prep_nodes)],
                input_id_set[len(num_in_prep_nodes) :],
            )

            raw_behavior[:, i] += probs_qnode(settings)

        return postmap @ raw_behavior if has_postmap else raw_behavior

    return behavior


def shannon_entropy(probs):
    """Evaluates the Shannon entropy for the given marginal probability
    distribution ``probs``.

    .. math::

        H(X) = -\\sum_{x\\in X}P(x)\\log_2(P(x))

    :param probs: A normalized probability vector.
    :type probs: np.array

    :returns: The Shannon entropy.
    :rtype: float
    """
    return -(
        math.sum([px * math.log2(px) if px != 0 and not (np.isclose(px, 0)) else 0 for px in probs])
    )
