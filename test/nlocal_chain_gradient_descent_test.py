import pytest
import pennylane as qml
from pennylane import numpy as np

from context import qnetvo as qnet


class TestNLocalChainGradientDescent:
    def RY_CNOT(self, settings, wires):
        qml.RY(settings[0], wires=wires[0])
        qml.RY(settings[1], wires=wires[1])
        qml.CNOT(wires=wires[0:2])

    @pytest.fixture
    def bilocal_chain_ansatz(self):
        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], self.RY_CNOT, 2),
            qnet.PrepareNode(1, [2, 3], self.RY_CNOT, 2),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
        ]

        return qnet.NetworkAnsatz(prep_nodes, meas_nodes)

    @pytest.fixture
    def optimization_args(self, bilocal_chain_ansatz):

        bilocal_chain_cost = qnet.nlocal_chain_cost_22(bilocal_chain_ansatz)

        np.random.seed(9)
        init_settings = bilocal_chain_ansatz.rand_scenario_settings()

        return bilocal_chain_cost, init_settings

    def test_bilocal_chain_gradient_descent(self, optimization_args):

        opt_dict = qnet.gradient_descent(
            *optimization_args, num_steps=10, step_size=2, sample_width=10, verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=1e-3)

    def test_bilocal_chain_parallel_gradient_descent(self, bilocal_chain_ansatz, optimization_args):

        parallel_grad = qnet.parallel_nlocal_chain_grad_fn(bilocal_chain_ansatz)

        opt_dict = qnet.gradient_descent(
            *optimization_args,
            num_steps=10,
            step_size=2,
            sample_width=10,
            grad_fn=parallel_grad,
            verbose=False
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=1e-3)

    def test_bilocal_chain_natural_gradient_descent(self):

        prep_nodes = [
            qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
        ]

        bilocal_chain_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)
        bilocal_chain_cost = qnet.nlocal_chain_cost_22(bilocal_chain_ansatz)

        np.random.seed(9)
        init_settings = bilocal_chain_ansatz.rand_scenario_settings()

        nat_grad = qnet.parallel_nlocal_chain_grad_fn(
            bilocal_chain_ansatz, natural_gradient=True, diff_method="parameter-shift"
        )

        opt_dict = qnet.gradient_descent(
            bilocal_chain_cost,
            init_settings,
            num_steps=8,
            step_size=1,
            sample_width=10,
            grad_fn=nat_grad,
            verbose=False,
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=1e-3)
