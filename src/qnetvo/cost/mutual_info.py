import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from ..qnodes import joint_probs_qnode
from ..information import shannon_entropy
from ..utilities import mixed_base_num


def mutual_info_cost_fn(
    ansatz, priors, postmap=np.array([]), static_layer="measure", **qnode_kwargs
):
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

    :param ansatz: The ansatz circuit on which the mutual information is evalutated.
    :type ansatz: NetworkAnsatz

    :param priors: A list of prior distributions for the inputs of each preparation node.
    :type priors: list[np.array]

    :param postmap: The post-processing matrix mapping the bitstring output from the
                    quantum device into the measurement node outputs.
    :type postmap: np.array

    :param static_layer: Either ``"prepare"`` or ``"measure"``, specifies which
                         network layer is held constant over all inputs.
    :type static_layer: String, default ``"measure"``.

    :param qnode_kwargs: Keyword arguments passed to the execute qnodes.
    :type qnode_kwargs: dictionary

    :returns: A cost function ``mutual_info_cost(scenario_settings)`` parameterized by
              the ansatz-specific scenario settings.
    :rtype: Function
    """

    num_prep_inputs = [prep_node.num_in for prep_node in ansatz.prepare_nodes]
    num_prep_nodes = len(ansatz.prepare_nodes)

    num_meas_inputs = [meas_node.num_in for meas_node in ansatz.measure_nodes]
    num_meas_nodes = len(ansatz.measure_nodes)

    num_inputs = num_prep_inputs if static_layer == "measure" else num_meas_inputs

    net_num_in = math.prod(num_inputs)
    node_input_ids = [mixed_base_num(i, num_inputs) for i in range(net_num_in)]

    net_num_out = math.prod([meas_node.num_out for meas_node in ansatz.measure_nodes])

    if len(postmap) == 0:
        postmap = np.eye(2 ** len(ansatz.measure_wires))

    px_vec = 1
    for i in range(len(priors)):
        px_vec = np.kron(px_vec, priors[i])

    Hx = shannon_entropy(px_vec)

    probs_qnode = joint_probs_qnode(ansatz, **qnode_kwargs)

    def cost_fn(scenario_settings):
        Hxy = 0
        py_vec = np.zeros(net_num_out)
        for (i, input_id_set) in enumerate(node_input_ids):

            if static_layer == "measure":
                prep_input_vals = input_id_set[0:num_prep_nodes]
                meas_input_vals = [0] * num_meas_nodes
            else:
                prep_input_vals = [0] * num_prep_nodes
                meas_input_vals = input_id_set[0:num_meas_nodes]

            settings = ansatz.qnode_settings(scenario_settings, prep_input_vals, meas_input_vals)

            p_mac = postmap @ probs_qnode(settings)

            Hxy += shannon_entropy(p_mac * px_vec[i])
            py_vec += p_mac * px_vec[i]

        Hy = shannon_entropy(py_vec)

        mutual_info = Hx + Hy - Hxy

        return -(mutual_info)

    return cost_fn
