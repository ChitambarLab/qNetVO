import pytest
import pennylane as qml
from pennylane import numpy as np

import qnetvo as qnet


def test_fixed_setting_optimization():
    prepare_nodes = [qnet.PrepareNode(2, [0], qnet.local_RY, 1)]
    measure_nodes = [qnet.MeasureNode(1, 2, [0], qnet.local_RY, 1)]
    ansatz = qnet.NetworkAnsatz(prepare_nodes, measure_nodes)

    np.random.seed(543)
    settings = ansatz.rand_network_settings(fixed_setting_ids=[0, 1], fixed_settings=[0, np.pi])
    cost = qnet.linear_probs_cost_fn(ansatz, np.eye(2))
    opt_dict = qnet.gradient_descent(
        cost, settings, step_size=1.5, num_steps=10, sample_width=1, verbose=True
    )

    assert np.isclose(opt_dict["opt_score"], 2)
    assert opt_dict["opt_settings"][0] == 0
    assert opt_dict["opt_settings"][1] == np.pi
    assert np.isclose(opt_dict["opt_settings"][2], 0, atol=1e-3)
