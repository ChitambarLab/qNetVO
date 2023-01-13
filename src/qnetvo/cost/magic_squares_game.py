from pennylane import math
from ..qnodes import joint_probs_qnode


def magic_squares_game_cost_fn(network_ansatz, **qnode_kwargs):
    """Constructs a cost function that maximizes the winning probability for the magic squares game.

    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :return: A cost function evaluated as ``cost(*network_settings)`` where
              the ``network_settings`` are obtained from the provided
              ``network_ansatz`` class.
    :rtype: Function
    """
    probs_qnode = joint_probs_qnode(network_ansatz, **qnode_kwargs)
    static_prep_inputs = [[0] * len(layer_nodes) for layer_nodes in network_ansatz.layers[0:-1]]

    def cost(*network_settings):
        winning_probability = 0
        for x in [0, 1, 2]:
            for y in [0, 1, 2]:
                settings = network_ansatz.qnode_settings(
                    network_settings, static_prep_inputs + [[x, y]]
                )
                probs = probs_qnode(settings)

                for i in range(16):
                    bit_string = [int(x) for x in math.binary_repr(i, 4)]

                    A_parity_bit = 0 if (bit_string[0] + bit_string[1]) % 2 == 0 else 1
                    B_parity_bit = 1 if (bit_string[2] + bit_string[3]) % 2 == 0 else 0

                    A_bits = bit_string[0:2] + [A_parity_bit]
                    B_bits = bit_string[2:] + [B_parity_bit]

                    if A_bits[y] == B_bits[x]:
                        winning_probability += probs[i]

        return -(winning_probability / 9)

    return cost
