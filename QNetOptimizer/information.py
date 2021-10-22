from pennylane import math
from .cost.qnodes import joint_probs_qnode
from .utilities import mixed_base_num
from pennylane import numpy as np


def behavior(network_ansatz, post_processing_map=np.array([]), qnode_kwargs={}):
    """A factory that constructs network-specific functions that use qnodes to
    collect the complete set of conditional probabilities for a given set of
    preparation and measurement settings.

    A network behavior is a column stochastic matrix containing the conditional
    probabilities for the network:

    .. math::

        \\mathbf{P}_{Net} = \\sum_{x,y,z}P(z|x,y)|z\\rangle\\langle x,y|

    :param network_ansatz: A class describing the particular quantum network.
    :type network_ansatz: NetworkAnsatz

    :param post_processing_map: A matrix describing how the :math:`2^{\\text{num wires}}`
                                outputs are mapped to the appropriate number of outputs.
    :type post_processing_map: np.ndarray

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

    if raw_net_num_out != net_num_out:
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

    node_input_ids = [mixed_base_num(i, base_digits) for i in range(net_num_in)]

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    def P_Net(scenario_settings):
        raw_behavior_matrix = np.zeros((raw_net_num_out, net_num_in))
        for (i, input_id_set) in enumerate(node_input_ids):
            settings = network_ansatz.qnode_settings(
                scenario_settings,
                input_id_set[0 : len(num_in_prep_nodes)],
                input_id_set[len(num_in_prep_nodes) :],
            )

            probs = probs_qnode(settings)
            raw_behavior_matrix[:, i] += probs

            if raw_net_num_out != net_num_out:
                behavior_matrix = post_processing_map @ raw_behavior_matrix
            else:
                behavior_matrix = raw_behavior_matrix

        return behavior_matrix

    return P_Net


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
    return -(math.sum([px * math.log2(px) if px != 0 else 0 for px in probs]))
