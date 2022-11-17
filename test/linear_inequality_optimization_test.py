import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo


def test_bipartite_guessing_game():
    entangled_prep_nodes = [qnetvo.PrepareNode(1, [0, 1], qnetvo.ghz_state, 0)]
    processing_nodes = [
        qnetvo.ProcessingNode(3, [0], qml.ArbitraryUnitary, 3),
        qnetvo.ProcessingNode(3, [1], qml.ArbitraryUnitary, 3),
    ]
    meas_nodes = [qnetvo.MeasureNode(1, 2, [0, 1], qml.ArbitraryUnitary, 15)]

    ansatz = qnetvo.NetworkAnsatz(entangled_prep_nodes, processing_nodes, meas_nodes)
    match = 7.5
    atol = 1e-2

    game = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 1, 1, 1, 0]])
    postmap = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    cost = qnetvo.linear_probs_cost_fn(ansatz, game, postmap=postmap)

    np.random.seed(14)
    settings = ansatz.rand_network_settings()

    opt_dict = qnetvo.gradient_descent(cost, settings, num_steps=25, step_size=0.2, sample_width=25)

    assert np.isclose(opt_dict["opt_score"], match, atol=atol)
