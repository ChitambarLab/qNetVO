from pennylane import math
from .cost.qnodes import joint_probs_qnode
from .utilities import mixed_base_num
from pennylane import numpy as np


def network_behavior_fn(network_ansatz, **qnode_kwargs):
    """A factory for constructing network-specific functions for constructing
    the network behavior for a given set of preparation and measurement settings.
    A network behavior is a column stochastic matrix containing the conditional
    probabilities for the network:

    .. math::

        \\mathbf{P}_{Net} = \\sum_{x,y,z}P(z|x,y)|z\\rangle\\langle x,y|

    :param network_ansatz: A class describing the particular quantum network.
    :type network_ansatz: NetworkAnsatz

    :returns: A function ``network_behavior(scenario_settings)`` that evaluates the
              behavior matrix for a given set of settings.
    :rtype: function
    """
    num_in_prep_nodes = [node.num_in for node in network_ansatz.prepare_nodes]
    num_in_meas_nodes = [node.num_in for node in network_ansatz.measure_nodes]

    base_digits = num_in_prep_nodes + num_in_meas_nodes
    net_num_in = math.prod(base_digits)
    net_num_out = 2 ** len(network_ansatz.measure_wires)

    node_input_ids = [mixed_base_num(i, base_digits) for i in range(net_num_in)]

    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)

    def network_behavior(scenario_settings):
        net_behavior = np.zeros((net_num_out, net_num_in))
        for (i, input_id_set) in enumerate(node_input_ids):
            prep_layer_settings = network_ansatz.layer_settings(
                scenario_settings[0], input_id_set[0 : len(num_in_prep_nodes)]
            )
            meas_layer_settings = network_ansatz.layer_settings(
                scenario_settings[1], input_id_set[len(num_in_prep_nodes) :]
            )

            probs = probs_qnode(prep_layer_settings, meas_layer_settings)
            net_behavior[:, i] += probs

        return net_behavior

    return network_behavior


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


def bisender_mac_mutual_info(mac_behavior, priors_x, priors_y):
    """Evaluates the rates and mutual information for the given
    conditional probability distribution ``mac_behavior`` and the
    corresponding input and output prior distributions.
    The mutual information for a bisender multiple access channel is
    characterized by three quantities:

    .. math::

        I(X;Z|Y) &= H(XY) + H(YZ) - H(Y) - H(XYZ) \\\\
        I(Y;Z|X) &= H(XY) + H(XZ) - H(X) - H(XYZ) \\\\
        I(XY;Z) &= H(XY) + H(Z) - H(XYZ)

    
    where :math:`H(X)` is the shannon entropy (see ``shannon_entropy``).

    :param mac_behavior: A column stochastic matrix describing the conditional
        probabilities the bi-sender multiple access channel.
    :type mac_behavior: np.array

    :param priors_x: A discrete probability vector describing the input set X.
    :type priors_x: np.array

    :param priors_y: A discrete probability vector describing the input set Y.
    :type priors_y: np.array

    :returns: A tuple with three values ``(I(X;Z|Y), I(Y;Z|X), I(XY;Z))``.
    :rtype: tuple
    """
    num_z = mac_behavior.shape[0]
    num_x = len(priors_x)
    num_y = len(priors_y)

    # joint probability distributions
    p_xy = math.kron(priors_x, priors_y)
    p_xyz = mac_behavior * p_xy
    p_z = math.array([math.sum(row) for row in p_xyz])

    p_yz = math.zeros((num_z, num_y))
    for x in range(num_x):
        p_yz += p_xyz[:, x * num_y : (x + 1) * num_y]

    p_xz = math.zeros((num_z, num_x))
    for y in range(num_y):
        p_xz += p_xyz[:, y : num_x * num_y : num_y]

    # shannon entropies
    H_x = shannon_entropy(priors_x)
    H_y = shannon_entropy(priors_y)
    H_z = shannon_entropy(p_z)
    H_xy = shannon_entropy(p_xy)
    H_xz = shannon_entropy(p_xz.reshape(num_x * num_z))
    H_yz = shannon_entropy(p_yz.reshape(num_y * num_z))
    H_xyz = shannon_entropy(p_xyz.reshape(num_x * num_y * num_z))

    # I(X;Z|Y)
    I_x_zy = H_xy + H_yz - H_y - H_xyz

    # I(Y;Z|X)
    I_y_zx = H_xy + H_xz - H_x - H_xyz

    # I(XY;Z)
    I_xy_z = H_xy + H_z - H_xyz

    return I_x_zy, I_y_zx, I_xy_z
