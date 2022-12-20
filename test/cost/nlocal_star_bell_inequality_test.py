import pytest
from pennylane import numpy as np
import pennylane as qml


import qnetvo as qnet


class TestNlocalStar22CostFn:
    def bilocal_star_ry_ansatz(self):
        prep_nodes = [
            qnet.PrepareNode(1, [0, 2], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [1, 3], qnet.ghz_state, 0),
        ]

        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [2, 3], qnet.local_RY, 2),
        ]

        return qnet.NetworkAnsatz(prep_nodes, meas_nodes)

    def trilocal_star_ry_ansatz(self):
        prep_nodes = [
            qnet.PrepareNode(1, [0, 3], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [1, 4], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 5], qnet.ghz_state, 0),
        ]

        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [2], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [3, 4, 5], qnet.local_RY, 3),
        ]

        return qnet.NetworkAnsatz(prep_nodes, meas_nodes)

    @pytest.mark.parametrize("parallel_flag, nthreads", [(False, 4), (True, 4), (True, 3)])
    def test_bilocal_star_22_cost(self, parallel_flag, nthreads):
        bilocal_star_ansatz = self.bilocal_star_ry_ansatz()
        bilocal_22_cost = qnet.nlocal_star_22_cost_fn(
            bilocal_star_ansatz, parallel=parallel_flag, nthreads=nthreads
        )

        zero_settings = bilocal_star_ansatz.zero_network_settings()

        assert np.isclose(bilocal_22_cost(*zero_settings), -1)

        ideal_settings = [0, np.pi / 2, np.pi / 4, -np.pi / 4, np.pi / 4, 0, -np.pi / 4, np.pi / 2]

        assert np.isclose(bilocal_22_cost(*ideal_settings), -(np.sqrt(2)))

    @pytest.mark.parametrize("parallel_flag, nthreads", [(False, 4), (True, 4), (True, 5)])
    def test_trilocal_star_cost(self, parallel_flag, nthreads):
        trilocal_star_ansatz = self.trilocal_star_ry_ansatz()
        trilocal_22_cost = qnet.nlocal_star_22_cost_fn(
            trilocal_star_ansatz,
            parallel=parallel_flag,
            nthreads=nthreads,
        )

        zero_settings = trilocal_star_ansatz.zero_network_settings()

        assert np.isclose(trilocal_22_cost(*zero_settings), -1)

        ideal_settings = [
            np.pi / 4,
            -np.pi / 4,
            np.pi / 4,
            -np.pi / 4,
            np.pi / 4,
            -np.pi / 4,
            0,
            0,
            0,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

        assert np.isclose(trilocal_22_cost(*ideal_settings), -np.sqrt(2))

    @pytest.mark.parametrize("parallel_flag, nthreads", [(False, 4), (True, 4), (True, 5)])
    def test_bilocal_star_22_cost_gradient_descent(self, parallel_flag, nthreads):
        bilocal_star_ansatz = self.bilocal_star_ry_ansatz()

        np.random.seed(45)
        opt_dict = qnet.gradient_descent(
            qnet.nlocal_star_22_cost_fn(
                bilocal_star_ansatz, parallel=parallel_flag, nthreads=nthreads
            ),
            bilocal_star_ansatz.rand_network_settings(),
            num_steps=10,
            step_size=2,
            sample_width=10,
            grad_fn=qnet.parallel_nlocal_star_grad_fn(bilocal_star_ansatz)
            if parallel_flag
            else None,
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=0.0001)

    @pytest.mark.parametrize("parallel_flag, nthreads", [(False, 4), (True, 4), (True, 5)])
    def test_trilocal_star_22_cost_gradient_descent(self, parallel_flag, nthreads):
        trilocal_star_ansatz = self.trilocal_star_ry_ansatz()

        np.random.seed(45)
        opt_dict = qnet.gradient_descent(
            qnet.nlocal_star_22_cost_fn(
                trilocal_star_ansatz, parallel=parallel_flag, nthreads=nthreads
            ),
            trilocal_star_ansatz.rand_network_settings(),
            num_steps=8,
            step_size=2,
            sample_width=10,
            grad_fn=qnet.parallel_nlocal_star_grad_fn(trilocal_star_ansatz)
            if parallel_flag
            else None,
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=0.0001)

    @pytest.mark.parametrize("nthreads", [3, 4])
    def test_bilocal_star_22_cost_natural_gradient_descent(self, nthreads):
        bilocal_star_ansatz = self.bilocal_star_ry_ansatz()

        np.random.seed(45)
        opt_dict = qnet.gradient_descent(
            qnet.nlocal_star_22_cost_fn(bilocal_star_ansatz, parallel=True, nthreads=nthreads),
            bilocal_star_ansatz.rand_network_settings(),
            num_steps=10,
            step_size=1.5,
            sample_width=1,
            grad_fn=qnet.parallel_nlocal_star_grad_fn(
                bilocal_star_ansatz, nthreads=nthreads, natural_gradient=True
            ),
        )

        assert np.isclose(opt_dict["opt_score"], np.sqrt(2), atol=1e-4)
