import pytest
import pennylane as qml
from pennylane import numpy as np

from context import QNetOptimizer as QNopt


class TestNoiseNode:
    def test_init(self):
        def circuit(settings, wires=[0, 1]):
            qml.AmplitudeDamping(0.7, wires=[0])
            qml.AmplitudeDamping(0.3, wires=[1])

        noise_node = QNopt.NoiseNode([0, 1], circuit)

        assert noise_node.wires == [0, 1]
        assert noise_node.ansatz_fn == circuit


class TestPrepareNode:
    def test_init(self):
        def circuit(settings, wires=[0, 1]):
            qml.RY(settings[0], wires=wires[0])
            qml.RY(settings[1], wires=wires[1])

        prep_node = QNopt.PrepareNode(3, [0, 1], circuit, 2)

        assert prep_node.num_in == 3
        assert prep_node.wires == [0, 1]
        assert prep_node.ansatz_fn == circuit
        assert prep_node.num_settings == 2
        assert prep_node.settings_dims == (3, 2)


class TestMeasureNode:
    def test_init(self):
        def circuit(settings, wires=[0, 1]):
            qml.RY(settings[0], wires=wires[0])
            qml.RY(settings[1], wires=wires[1])

        measure_node = QNopt.MeasureNode(3, 4, [0, 1], circuit, 2)

        assert measure_node.num_in == 3
        assert measure_node.num_out == 4
        assert measure_node.wires == [0, 1]
        assert measure_node.ansatz_fn == circuit
        assert measure_node.num_settings == 2
        assert measure_node.settings_dims == (3, 2)


class TestNetworkAnsatz:

    def chsh_ansatz(self):
        prepare_nodes = [
            QNopt.PrepareNode(1,[0,1],QNopt.local_RY, 2)
        ]
        measure_nodes = [
            QNopt.MeasureNode(2,2,[0], QNopt.local_RY, 1),
            QNopt.MeasureNode(2,2,[1], QNopt.local_RY, 1)
        ]

        return QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)

    def test_init(self):
        # setup test
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        def noisy_ansatz_circuit(settings, wires):
            qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])

        prepare_nodes = [
            QNopt.PrepareNode(1, [0], ansatz_circuit, 1),
            QNopt.PrepareNode(1, [1], ansatz_circuit, 1),
            QNopt.PrepareNode(1, [2], ansatz_circuit, 1),
        ]
        measure_nodes = [
            QNopt.MeasureNode(1, 2, [0], ansatz_circuit, 1),
            QNopt.MeasureNode(1, 2, [1], ansatz_circuit, 1),
        ]
        noise_nodes = [
            QNopt.NoiseNode([1], noisy_ansatz_circuit),
            QNopt.NoiseNode([2], noisy_ansatz_circuit),
        ]

        network_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)
        noisy_network_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes, noise_nodes)

        # verify network nodes
        assert network_ansatz.prepare_nodes == prepare_nodes
        assert network_ansatz.measure_nodes == measure_nodes
        assert network_ansatz.noise_nodes == []

        # verify wires
        assert network_ansatz.prepare_wires.tolist() == [0, 1, 2]
        assert network_ansatz.measure_wires.tolist() == [0, 1]
        assert network_ansatz.noise_wires.tolist() == []
        assert network_ansatz.network_wires.tolist() == [0, 1, 2]

        # verify device
        assert network_ansatz.default_dev_name == "default.qubit"
        assert network_ansatz.dev.wires.tolist() == [0, 1, 2]
        assert network_ansatz.dev.short_name == "default.qubit"

        # verify qnode construction and execution
        @qml.qnode(network_ansatz.dev)
        def test_circuit(prepare_settings_array, measure_settings_array):
            network_ansatz.fn(prepare_settings_array, measure_settings_array)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert test_circuit([[0], [0], [0]], [[0], [0]]) == 1
        assert test_circuit([[np.pi / 4], [-np.pi / 3], [0]], [[-np.pi / 4], [np.pi / 3]]) == 1

        # Noisy network Case
        noisy_network_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes, noise_nodes)

        # verify network nodes
        assert noisy_network_ansatz.prepare_nodes == prepare_nodes
        assert noisy_network_ansatz.measure_nodes == measure_nodes
        assert noisy_network_ansatz.noise_nodes == noise_nodes

        # verify wires
        assert noisy_network_ansatz.prepare_wires.tolist() == [0, 1, 2]
        assert noisy_network_ansatz.measure_wires.tolist() == [0, 1]
        assert noisy_network_ansatz.noise_wires.tolist() == [1, 2]
        assert noisy_network_ansatz.network_wires.tolist() == [0, 1, 2]

        # verify device
        assert noisy_network_ansatz.dev.wires.tolist() == [0, 1, 2]
        assert noisy_network_ansatz.dev.short_name == "default.mixed"

        # verify qnode construction and execution
        @qml.qnode(noisy_network_ansatz.dev)
        def noisy_test_circuit(prepare_settings_array, measure_settings_array):
            noisy_network_ansatz.fn(prepare_settings_array, measure_settings_array)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert noisy_test_circuit([[0], [0], [0]], [[0], [0]]) == 0.5
        assert np.isclose(
            noisy_test_circuit([[np.pi / 4], [-np.pi / 3], [0]], [[-np.pi / 4], [np.pi / 3]]), 0.5
        )

    def test_layer_settings(self):
        nodes = [
            QNopt.PrepareNode(3, [0], QNopt.local_RY, 1),
            QNopt.PrepareNode(3, [1], QNopt.local_RY, 1),
            QNopt.PrepareNode(3, [2], QNopt.local_RY, 1),
            QNopt.PrepareNode(3, [3], QNopt.local_RY, 1),
        ]

        ansatz = QNopt.NetworkAnsatz(nodes, nodes)

        # test
        np.random.seed(123)
        scenario_settings = ansatz.rand_scenario_settings()

        layer_settings = ansatz.layer_settings(scenario_settings[0], [0, 1, 2, 1])

        assert len(layer_settings) == 4
        assert np.isclose(layer_settings[0], 1.2344523)
        assert np.isclose(layer_settings[1], 1.37896421)
        assert np.isclose(layer_settings[2], -0.1198084)
        assert np.isclose(layer_settings[3], -0.98534158)

    def test_network_ansatz_device(self):
        
        chsh_ansatz = self.chsh_ansatz()
        assert chsh_ansatz.dev.short_name == "default.qubit"

        updated_dev = chsh_ansatz.device("default.mixed", shots=5)

        assert updated_dev == chsh_ansatz.dev
        assert chsh_ansatz.dev.short_name == "default.mixed"
        assert chsh_ansatz.dev.shots == 5

    def test_circuit_layer(self):
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        def noisy_ansatz_circuit(settings, wires):
            qml.Hadamard(wires=wires[0])
            qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])

        node1 = QNopt.PrepareNode(1, [0], ansatz_circuit, 1)
        node2 = QNopt.PrepareNode(1, [1], ansatz_circuit, 1)

        noisy_node1 = QNopt.NoiseNode([0], noisy_ansatz_circuit)
        noisy_node2 = QNopt.NoiseNode([1], noisy_ansatz_circuit)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def test_circuit(settings):
            QNopt.NetworkAnsatz.circuit_layer([node1, node2])(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert test_circuit([[0], [0]]) == 1
        assert test_circuit([[np.pi], [0]]) == -1

        val = test_circuit([[np.pi / 4], [-np.pi / 4]])
        assert np.isclose(val, 0.5)

        @qml.qnode(qml.device("default.mixed", wires=2))
        def noisy_test_circuit(settings):
            QNopt.NetworkAnsatz.circuit_layer([noisy_node1, noisy_node2])(settings)

            return qml.state()

        assert np.allclose(
            noisy_test_circuit([[], []]),
            [
                [0.25, 0.125, 0.125, 0.0625],
                [0.125, 0.25, 0.0625, 0.125],
                [0.125, 0.0625, 0.25, 0.125],
                [0.0625, 0.125, 0.125, 0.25],
            ],
        )

    def test_collect_wires(self):
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        node1 = QNopt.MeasureNode(1, 2, [1], ansatz_circuit, 1)
        node2 = QNopt.MeasureNode(1, 2, [0], ansatz_circuit, 1)

        ansatz_wires = QNopt.NetworkAnsatz.collect_wires([node1, node2])
        assert ansatz_wires.tolist() == [1, 0]

        node3 = QNopt.MeasureNode(1, 2, [0], ansatz_circuit, 1)
        with pytest.raises(
            ValueError,
            match="One or more wires are not unique. Each node must contain unique wires.",
        ):
            QNopt.NetworkAnsatz.collect_wires([node2, node3])

    def test_scenario_settings(self):
        def ansatz_circuit(settings, wires):
            return None

        prep_nodes = [
            QNopt.PrepareNode(3, [0], ansatz_circuit, 2),
            QNopt.PrepareNode(2, [1], ansatz_circuit, 4),
        ]
        meas_nodes = [
            QNopt.MeasureNode(2, 2, [0], ansatz_circuit, 1),
            QNopt.MeasureNode(1, 2, [1], ansatz_circuit, 3),
        ]

        network_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

        zero_settings = network_ansatz.zero_scenario_settings()

        match_settings = [
            [[[0, 0], [0, 0], [0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]],
            [[[0], [0]], [[0, 0, 0]]],
        ]

        assert len(zero_settings[0]) == 2
        assert np.array_equal(zero_settings[0][0], match_settings[0][0])
        assert np.array_equal(zero_settings[0][1], match_settings[0][1])

        assert len(zero_settings[1]) == 2
        assert np.array_equal(zero_settings[1][0], match_settings[1][0])
        assert np.array_equal(zero_settings[1][1], match_settings[1][1])

        np.random.seed(123)
        rand_settings = network_ansatz.rand_scenario_settings()

        match_settings = [
            [
                [[1.2344523, -1.34372619], [-1.71624293, 0.3224202], [1.37896421, -0.48313636]],
                [
                    [3.02073055, 1.1613195, -0.1198084, -0.67784562],
                    [-0.98534158, 1.43916176, -0.38596197, -2.76662537],
                ],
            ],
            [[[-0.64060684], [1.49536924]], [[-1.99496329, -2.03919676, 0.19824313]]],
        ]

        assert len(rand_settings[0]) == 2
        assert np.allclose(rand_settings[0][0], match_settings[0][0])
        assert np.allclose(rand_settings[0][1], match_settings[0][1])

        assert len(rand_settings[1]) == 2
        assert np.allclose(rand_settings[1][0], match_settings[1][0])
        assert np.allclose(rand_settings[1][1], match_settings[1][1])
