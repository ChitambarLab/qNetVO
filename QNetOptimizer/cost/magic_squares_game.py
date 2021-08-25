import pennylane.numpy as np
from .qnodes import joint_probs_qnode

def magic_squares_game_cost(network_ansatz):
    """Constructs a cost function that maximizes the winning probability for the magic squares game.
    
    :param network_ansatz: A ``NetworkAnsatz`` class specifying the quantum network simulation.
    :type network_ansatz: NetworkAnsatz

    :return: A cost function evaluated as ``cost(scenario_settings)`` where
              the ``scenario_settings`` are obtained from the provided
              ``network_ansatz`` class.
    :rtype: Function
    """
    probs_qnode = joint_probs_qnode(network_ansatz)

    def cost(settings):
        state_settings, measure_settings = settings
        A_settings, B_settings = measure_settings

        winning_probability = 0
        for x in [0,1,2]:
            for y in [0,1,2]:
                probs = probs_qnode(state_settings[0], [A_settings[x], B_settings[y]])

                for i in range(16):
                    bit_string = [int(x) for x in np.binary_repr(i,4)]

                    A_parity_bit = 0 if (bit_string[0] + bit_string[1]) % 2 == 0 else 1
                    B_parity_bit = 1 if (bit_string[2] + bit_string[3]) % 2 == 0 else 0

                    A_bits = bit_string[0:2] + [A_parity_bit]
                    B_bits = bit_string[2:] + [B_parity_bit]            

                    if A_bits[y] == B_bits[x]:
                        winning_probability += probs[i]

        return -(winning_probability/9)
    
    return cost