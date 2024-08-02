import pytest
from pennylane import numpy as np

import qnetvo as qnet


class TestNegativityCost:
    def test_negativity_cost_fn(self):

        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], qnet.bell_state_copies, 2),
        ]

        ansatz = qnet.NetworkAnsatz(prep_nodes)

        negativity_cost = qnet.negativity_cost_fn(ansatz, m=1, n=1, wires=[0, 1])

        zero_settings = ansatz.zero_network_settings()

        negativity_value = negativity_cost(*zero_settings)

        expected_negativity = -0.5
        assert np.isclose(
            negativity_value, expected_negativity
        ), f"Expected {expected_negativity}, but got {negativity_value}"

        separable_prep_nodes = [
            qnet.PrepareNode(4, [0, 1], qnet.local_RY, 2),
        ]

        separable_ansatz = qnet.NetworkAnsatz(separable_prep_nodes)

        separable_negativity_cost = qnet.negativity_cost_fn(
            separable_ansatz, m=1, n=1, wires=[0, 1]
        )

        separable_negativity_value = separable_negativity_cost(*zero_settings)

        expected_separable_negativity = 0
        assert np.isclose(
            separable_negativity_value, expected_separable_negativity
        ), f"Expected {expected_separable_negativity}, but got {separable_negativity_value}"

        with pytest.raises(ValueError, match="Sum of sizes of two subsystems should be"):
            qnet.negativity_cost_fn(ansatz, m=2, n=1, wires=[0, 1])
