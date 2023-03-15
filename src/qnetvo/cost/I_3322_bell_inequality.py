from pennylane import math
from ..postprocessing import even_parity_ids
from ..qnodes import joint_probs_qnode, local_parity_expval_qnode


def post_process_I_3322_joint_probs(probs_vec):
    """Applies post-processing to multi-qubit probabilities in order to coarse-grain
    them into the dichotomic parity observables required by the :math:`I_{3322}` inequality.

    An :math:`N`-qubit circuit has :math:`2^N` measurement outcomes.
    To construct the joint probabilitye :math:`P(00|xy)` for binary outputs, the joint
    probabilities can be partitioned into two sets, :math:`\\{Even\\}` and :math`\\{Odd\\}` which
    denote the set of *Even* and *Odd* parity bit strings.
    The :math:`2^N` joint probabilities are expressed as :math:`P(\\vec{a},\\vec{b}|x,y)` where
    :math:`\\vec{a}` and :math:`\\vec{b}` are each :math:`N`-bit strings.
    Since the :math:`I_{3322}` inequality only requires dichotomic probabilities :math:`P(00|xy)`,
    our post-processing only needs to calculate this value.
    To reduce the joint probabilities :math:`P(\\vec{a},\\vec{b}|x,y)` to dichotomic probabilities
    :math:`P(00|x,y)` we aggregate the probabilities of even parity bit strings with

    .. math::

       P(00|xy) = \\{Even\\}_A\\{Even\\}_B
       = \\sum_{\\vec{a}\\in \\{Even\\}} \\sum_{\\vec{b}\\in\\{Even\\}} P(\\vec{a},\\vec{b}|x,y).

    :param n_qubits: The number of wires measured to obtained the joint probabilites.
    :type n_qubits: int

    :param probs_vec: A probability vector obtained by measuring all wires in the
        computational basis.
    :type probs_vec: list[float]

    :returns: The dichotomic probability :math:`P(00|xy)`.
    """
    n_local_qubits = int(math.log2(len(probs_vec)) / 2)
    probs = math.reshape(probs_vec, (2**n_local_qubits, 2**n_local_qubits))
    even_ids = even_parity_ids(n_local_qubits)

    return sum([sum([probs[a, b] for b in even_ids]) for a in even_ids])


def I_3322_bell_inequality_cost_fn(network_ansatz, **qnode_kwargs):
    """Constructs a cost function that maximizes the score of the :math:`I_{3322}` Bell inequality.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :returns: A cost function evaluated as ``cost(*network_settings)`` where
              the ``network_settings`` are obtained from the provided
              ``network_ansatz`` class.
    """
    I_3322_joint_probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)
    I_3322_local_expval_qnode = local_parity_expval_qnode(network_ansatz, **qnode_kwargs)

    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]

    def cost(*network_settings):
        score = 0
        for x, y, mult in [
            (0, 0, 1),
            (0, 1, 1),
            (0, 2, 1),
            (1, 0, 1),
            (1, 1, 1),
            (1, 2, -1),
            (2, 0, 1),
            (2, 1, -1),
        ]:
            settings = network_ansatz.qnode_settings(
                network_settings, static_prep_inputs + [[x, y]]
            )

            probs_vec_xy = I_3322_joint_probs_qnode(settings)
            prob00_xy = post_process_I_3322_joint_probs(probs_vec_xy)

            score += mult * prob00_xy

        settings_00 = network_ansatz.qnode_settings(network_settings, static_prep_inputs + [[0, 0]])
        settings_11 = network_ansatz.qnode_settings(network_settings, static_prep_inputs + [[1, 1]])

        expval_00 = I_3322_local_expval_qnode(settings_00)
        expval_11 = I_3322_local_expval_qnode(settings_11)

        # - P_A(0|0)
        score += -1 * (expval_00[0] + 1) / 2

        # - 2 * P_B(0|0)
        score += -2 * (expval_00[1] + 1) / 2

        # - P_B(0|1)
        score += -1 * (expval_11[1] + 1) / 2

        return -(score)

    return cost
