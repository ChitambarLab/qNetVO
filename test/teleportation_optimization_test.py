import pytest
import pennylane as qml
import numpy as np
from scipy.stats import unitary_group


from pennylane import numpy as qnp

import qnetvo


class TestTeleportationOptimization:
    # prep node for each qubit state to teleport
    def input_prep_node(self, state_vec):
        def circuit(settings, wires):
            qml.QubitStateVector(state_vec, wires=wires[0])

        return qnetvo.PrepareNode(num_in=1, wires=[0], ansatz_fn=circuit, num_settings=0)

    # prep node for shared entanglement
    @property
    def ent_prep_node(self):
        return qnetvo.PrepareNode(
            num_in=1,
            wires=[1, 2],
            ansatz_fn=qml.ArbitraryStatePreparation,
            num_settings=6,
        )

    # locc measurement node
    @property
    def cc_sender_nodes(self):
        def locc_circuit(settings, wires):
            qml.CNOT(wires=wires[0:2])
            qml.Rot(*settings[0:3], wires=wires[0])

            b0 = qml.measure(wires[0])
            b1 = qml.measure(wires[1])

            return [b0, b1]

        return [
            qnetvo.CCSenderNode(
                num_in=1,
                wires=[0, 1],
                cc_wires_out=[0, 1],
                ansatz_fn=locc_circuit,
                num_settings=3,
            ),
        ]

    # teleportation output measurement node
    @property
    def cc_receiver_nodes(self):
        def measure_circuit(settings, wires, cc_wires):
            qml.cond((cc_wires[0] == 0) & (cc_wires[1] == 1), qml.Rot)(*settings[0:3], wires=[2])
            qml.cond((cc_wires[0] == 1) & (cc_wires[1] == 0), qml.Rot)(*settings[3:6], wires=[2])
            qml.cond((cc_wires[0] == 0) & (cc_wires[1] == 1), qml.Rot)(*settings[6:9], wires=[2])
            qml.cond((cc_wires[0] == 1) & (cc_wires[1] == 1), qml.Rot)(*settings[9:12], wires=[2])

        return [
            qnetvo.CCReceiverNode(
                num_in=1,
                wires=[2],
                cc_wires_in=[0, 1],
                ansatz_fn=measure_circuit,
                num_settings=12,
            )
        ]

    # constructs a cost function that trains a teleportation protocol
    def cost_fn(self, input_states):
        input_prep_nodes = [self.input_prep_node(state) for state in input_states]
        ansatzes = [
            qnetvo.NetworkAnsatz(
                [node, self.ent_prep_node], self.cc_sender_nodes, self.cc_receiver_nodes
            )
            for node in input_prep_nodes
        ]

        # construct a qnode that outputs a density matrix
        def teleport_qnode_fn(ansatz):
            @qml.qnode(qml.device(**ansatz.dev_kwargs))
            def teleport(settings):
                ansatz.fn(settings)
                return qml.density_matrix(2)

            return teleport

        teleport_circuits = [teleport_qnode_fn(ansatz) for ansatz in ansatzes]

        # minimize average teleportation fidelity
        def cost(*settings):
            cost_val = 0
            for i in range(len(teleport_circuits)):
                teleport_circuit = teleport_circuits[i]
                input_state = input_states[i]

                rho = teleport_circuit(settings)
                rho_target = np.outer(input_state, input_state.conj())

                cost_val -= qml.math.fidelity(rho, rho_target)

            return cost_val / len(teleport_circuits)

        return cost

    def test_teleportation_optimization(self):
        # setting up optimization
        training_states = [
            np.array([1, 0]),
            np.array([1, 1]) / np.sqrt(2),
            np.array([1, -1]) / np.sqrt(2),
            np.array([1, 1j]) / np.sqrt(2),
        ]

        training_cost = self.cost_fn(training_states)

        qnp.random.seed(49)
        init_settings = qnetvo.NetworkAnsatz(
            [self.input_prep_node(training_states[0]), self.ent_prep_node],
            self.cc_sender_nodes,
            self.cc_receiver_nodes,
        ).rand_network_settings()

        # optimizing teleportation protocol
        opt_dict = qnetvo.gradient_descent(
            training_cost,
            init_settings,
            step_size=3.1,
            num_steps=90,
            sample_width=90,
            verbose=False,
        )

        # generating new random states to test teleportation protocol
        test_states = [unitary_group.rvs(2)[:, 0] for i in range(100)]
        test_cost = self.cost_fn(test_states)

        assert np.isclose(test_cost(*opt_dict["opt_settings"]), -1, atol=5e-3)
