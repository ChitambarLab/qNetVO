import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from ..qnodes import joint_probs_qnode
from ..information import shannon_entropy
from ..utilities import mixed_base_num, ragged_reshape


def mutual_info_cost_fn(ansatz, priors, postmap=np.array([]), **qnode_kwargs):
    """Constructs an ansatz-specific mutual information cost function.

    The mutual information quantifies the information shared by two distributions
    :math:`X` and :math:`Y`.
    This entropic quantity is expressed as

    .. math::

            I(Y;X) = H(Y) + H(X) - H(XY)

    where :math:`H(X) = -\\sum_{i}P(X_i)\\log_2(P(X_i))` is the Shannon entropy
    (see :meth:`qnetvo.shannon_entropy`).

    The mutual information can be used to quantify the amount of communication between
    a sender and receiver.
    In a quantum prepare and measure network, we evaluate the mutual information between
    the collections of preparation and measurement nodes.

    :param ansatz: The ansatz circuit on which the mutual information is evaluated.
    :type ansatz: NetworkAnsatz

    :param priors: A list of prior distributions for the inputs of each preparation node.
    :type priors: list[np.array]

    :param postmap: The post-processing matrix mapping the bitstring output from the
                    quantum device into the measurement node outputs.
    :type postmap: np.array

    :param qnode_kwargs: Keyword arguments passed to the execute qnodes.
    :type qnode_kwargs: dictionary

    :returns: A cost function ``mutual_info_cost(*network_settings)`` parameterized by
              the ansatz-specific scenario settings.
    :rtype: Function
    """

    net_num_in = math.prod(ansatz.layers_total_num_in)
    num_inputs_list = math.concatenate(ansatz.layers_node_num_in).tolist()
    node_input_ids = [
        ragged_reshape(mixed_base_num(i, num_inputs_list), ansatz.layers_num_nodes)
        for i in range(net_num_in)
    ]

    net_num_out = math.prod([meas_node.num_out for meas_node in ansatz.layers[-1]])

    if len(postmap) == 0:
        postmap = np.eye(2 ** len(ansatz.layers_wires[-1]))

    px_vec = 1
    for i in range(len(priors)):
        px_vec = np.kron(px_vec, priors[i])

    Hx = shannon_entropy(px_vec)

    probs_qnode = joint_probs_qnode(ansatz, **qnode_kwargs)

    def cost(*network_settings):
        Hxy = 0
        py_vec = np.zeros(net_num_out)
        for i, input_id_set in enumerate(node_input_ids):
            settings = ansatz.qnode_settings(network_settings, input_id_set)
            p_net = postmap @ probs_qnode(settings)

            Hxy += shannon_entropy(p_net * px_vec[i])
            py_vec += p_net * px_vec[i]

        Hy = shannon_entropy(py_vec)

        mutual_info = Hx + Hy - Hxy

        return -(mutual_info)

    return cost


def shannon_entropy_cost_fn(ansatz, **qnode_kwargs):
    """Constructs an ansatz-specific Shannon entropy cost function

    The Shannon entropy characterizes the amount of randomness, or similarly, the amount of
    information is present in a random variable. Formally, let :math:`X` be a discrete random
    variable, then the Shannon entropy is defined by the expression:

    .. math::

            H(X) = -\\sum_{x} P(x) \\log_{2} P(x)

    In the case of a quantum network, the Shannon entropy is defined on the measurement outcome
    of the network ansatz.

    :param ansatz: The ansatz circuit on which the Shannon entropy is evalutated.
    :type ansatz: NetworkAnsatz

    :param qnode_kwargs: Keyword arguments passed to the execute qnodes.
    :type qnode_kwargs: dictionary

    :returns: A cost function ``shannon_entropy(*network_settings)`` parameterized by
              the ansatz-specific network settings.
    :rtype: Function
    """
    static_inputs = [[0] * num_nodes for num_nodes in ansatz.layers_num_nodes]
    probs_qnode = joint_probs_qnode(ansatz, **qnode_kwargs)

    def cost(*network_settings):
        settings = ansatz.qnode_settings(network_settings, static_inputs)
        probs_vec = probs_qnode(settings)

        return shannon_entropy(probs_vec)

    return cost
