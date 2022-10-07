import pytest
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf

import qnetvo as qnet


class TestNoiseNode:
    def test_init(self):
        def circuit(settings, wires=[0, 1]):
            qml.AmplitudeDamping(0.7, wires=[0])
            qml.AmplitudeDamping(0.3, wires=[1])

        noise_node = qnet.NoiseNode([0, 1], circuit)

        assert noise_node.wires == [0, 1]
        assert noise_node.ansatz_fn == circuit


class TestPrepareNode:
    def test_init(self):
        def circuit(settings, wires=[0, 1]):
            qml.RY(settings[0], wires=wires[0])
            qml.RY(settings[1], wires=wires[1])

        prep_node = qnet.PrepareNode(3, [0, 1], circuit, 2)

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

        measure_node = qnet.MeasureNode(3, 4, [0, 1], circuit, 2)

        assert measure_node.num_in == 3
        assert measure_node.num_out == 4
        assert measure_node.wires == [0, 1]
        assert measure_node.ansatz_fn == circuit
        assert measure_node.num_settings == 2
        assert measure_node.settings_dims == (3, 2)


class TestNetworkAnsatz:
    def chsh_ansatz(self):
        prepare_nodes = [qnet.PrepareNode(1, [0, 1], qnet.local_RY, 2)]
        measure_nodes = [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
        ]

        return qnet.NetworkAnsatz(prepare_nodes, measure_nodes)

    def test_init(self):
        # setup test
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        def noisy_ansatz_circuit(settings, wires):
            qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])

        prepare_nodes = [
            qnet.PrepareNode(1, [0], ansatz_circuit, 1),
            qnet.PrepareNode(1, [1], ansatz_circuit, 1),
            qnet.PrepareNode(1, [2], ansatz_circuit, 1),
        ]
        measure_nodes = [
            qnet.MeasureNode(1, 2, [0], ansatz_circuit, 1),
            qnet.MeasureNode(1, 2, [1], ansatz_circuit, 1),
        ]
        noise_nodes = [
            qnet.NoiseNode([1], noisy_ansatz_circuit),
            qnet.NoiseNode([2], noisy_ansatz_circuit),
        ]

        network_ansatz = qnet.NetworkAnsatz(prepare_nodes, measure_nodes)
        noisy_network_ansatz = qnet.NetworkAnsatz(prepare_nodes, measure_nodes, noise_nodes)

        # verify network nodes
        assert network_ansatz.prepare_nodes == prepare_nodes
        assert network_ansatz.measure_nodes == measure_nodes
        assert network_ansatz.noise_nodes == []

        # verify network settings partitions
        assert network_ansatz.parameter_partitions == [
            [[(0, 1)], [(1, 2)], [(2, 3)]],
            [[(3, 4)], [(4, 5)]],
        ]

        # verify wires
        assert network_ansatz.prepare_wires.tolist() == [0, 1, 2]
        assert network_ansatz.measure_wires.tolist() == [0, 1]
        assert network_ansatz.noise_wires.tolist() == []
        assert network_ansatz.network_wires.tolist() == [0, 1, 2]

        # verify device
        assert network_ansatz.dev_kwargs["name"] == "default.qubit"
        assert network_ansatz.dev.wires.tolist() == [0, 1, 2]
        assert network_ansatz.dev.short_name == "default.qubit"

        # verify qnode construction and execution
        @qml.qnode(network_ansatz.dev)
        def test_circuit(settings):
            network_ansatz.fn(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.isclose(test_circuit([0, 0, 0, 0, 0]), 1)
        assert np.isclose(test_circuit([np.pi / 4, -np.pi / 3, 0, -np.pi / 4, np.pi / 3]), 1)

        # Noisy network Case
        noisy_network_ansatz = qnet.NetworkAnsatz(prepare_nodes, measure_nodes, noise_nodes)

        # verify network nodes
        assert noisy_network_ansatz.prepare_nodes == prepare_nodes
        assert noisy_network_ansatz.measure_nodes == measure_nodes
        assert noisy_network_ansatz.noise_nodes == noise_nodes

        # verify network settings partitions
        assert network_ansatz.parameter_partitions == [
            [[(0, 1)], [(1, 2)], [(2, 3)]],
            [[(3, 4)], [(4, 5)]],
        ]

        # verify wires
        assert noisy_network_ansatz.prepare_wires.tolist() == [0, 1, 2]
        assert noisy_network_ansatz.measure_wires.tolist() == [0, 1]
        assert noisy_network_ansatz.noise_wires.tolist() == [1, 2]
        assert noisy_network_ansatz.network_wires.tolist() == [0, 1, 2]

        # verify device
        assert noisy_network_ansatz.dev_kwargs["name"] == "default.mixed"
        assert noisy_network_ansatz.dev.wires.tolist() == [0, 1, 2]
        assert noisy_network_ansatz.dev.short_name == "default.mixed"

        # verify qnode construction and execution
        @qml.qnode(noisy_network_ansatz.dev)
        def noisy_test_circuit(settings):
            noisy_network_ansatz.fn(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.isclose(noisy_test_circuit([0, 0, 0, 0, 0]), 0.5)
        assert np.isclose(
            noisy_test_circuit([np.pi / 4, -np.pi / 3, 0, -np.pi / 4, np.pi / 3]), 0.5
        )

    def test_init_noisy_device_override(self):
        prep_nodes = [qnet.PrepareNode(1, [0], qnet.local_RY, 1)]
        noise_nodes = [
            qnet.NoiseNode(
                [0, 1], lambda settings, wires: qnet.pure_amplitude_damping([0.5], wire=wires)
            )
        ]
        meas_nodes = [qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1)]

        mixed_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes, noise_nodes)

        assert mixed_ansatz.dev.short_name == "default.mixed"

        pure_ansatz = qnet.NetworkAnsatz(
            prep_nodes, meas_nodes, noise_nodes, dev_kwargs={"name": "default.qubit"}
        )

        assert pure_ansatz.dev.short_name == "default.qubit"

    def test_partition_settings_slices(self):
        prep_nodes = [
            qnet.PrepareNode(3, [0, 1], qml.ArbitraryStatePreparation, 6),
            qnet.PrepareNode(2, [2], qnet.local_RY, 1),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [1], qml.ArbitraryUnitary, 3),
            qnet.MeasureNode(2, 4, [0, 2], qml.ArbitraryUnitary, 15),
        ]

        network_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        assert network_ansatz.get_network_parameter_partitions() == [
            [[(0, 6), (6, 12), (12, 18)], [(18, 19), (19, 20)]],
            [[(20, 23), (23, 26)], [(26, 41), (41, 56)]],
        ]

    def test_qnode_settings(self):
        chsh_ansatz = self.chsh_ansatz()

        np.random.seed(123)
        network_settings = chsh_ansatz.rand_network_settings()

        settings = chsh_ansatz.qnode_settings(network_settings, [[0], [0, 1]])

        assert np.allclose(settings, [1.2344523, -1.34372619, -1.71624293, -0.48313636])

    @pytest.mark.parametrize(
        "layer_inputs, layer_id, match",
        [([1, 0], 0, [1, 2]), ([0, 1], 0, [0, 3]), ([1], 1, [6, 7])],
    )
    def test_layer_settings(self, layer_inputs, layer_id, match):
        network_settings = [
            np.array(0),
            np.array(1),
            np.array(2),
            np.array(3),
            np.array(4),
            np.array(5),
            np.array(6),
            np.array(7),
        ]
        prep_nodes = [
            qnet.PrepareNode(2, [0], qnet.local_RY, 1),
            qnet.PrepareNode(2, [1], qnet.local_RY, 1),
        ]
        meas_nodes = [qnet.MeasureNode(2, 2, [0, 1], qnet.local_RY, 2)]
        network_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        settings = network_ansatz.layer_settings(network_settings, layer_id, layer_inputs)
        assert np.allclose(settings, match)

    def test_network_ansatz_device(self):

        chsh_ansatz = self.chsh_ansatz()
        assert chsh_ansatz.dev.short_name == "default.qubit"
        assert chsh_ansatz.dev_kwargs["name"] == "default.qubit"

        updated_dev = chsh_ansatz.set_device("default.mixed", shots=5)
        assert updated_dev == chsh_ansatz.dev
        assert chsh_ansatz.dev.short_name == "default.mixed"
        assert chsh_ansatz.dev_kwargs["name"] == "default.mixed"
        assert chsh_ansatz.dev.shots == 5
        assert chsh_ansatz.dev_kwargs["shots"] == 5

        # device() instantiates a new device
        dev1 = chsh_ansatz.device()
        dev2 = chsh_ansatz.device()

        assert dev1 != dev2
        assert dev1.short_name == dev2.short_name
        assert dev1.shots == dev2.shots

    def test_circuit_layer(self):
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        def noisy_ansatz_circuit(settings, wires):
            qml.Hadamard(wires=wires[0])
            qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])

        node1 = qnet.PrepareNode(1, [0], ansatz_circuit, 1)
        node2 = qnet.PrepareNode(1, [1], ansatz_circuit, 1)

        noisy_node1 = qnet.NoiseNode([0], noisy_ansatz_circuit)
        noisy_node2 = qnet.NoiseNode([1], noisy_ansatz_circuit)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def test_circuit(settings):
            qnet.NetworkAnsatz.circuit_layer([node1, node2])(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.isclose(test_circuit([0, 0]), 1)
        assert np.isclose(test_circuit([np.pi, 0]), -1)

        val = test_circuit([np.pi / 4, -np.pi / 4])
        assert np.isclose(val, 0.5)

        @qml.qnode(qml.device("default.mixed", wires=2))
        def noisy_test_circuit(settings):
            qnet.NetworkAnsatz.circuit_layer([noisy_node1, noisy_node2])(settings)

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

        node1 = qnet.MeasureNode(1, 2, [1], ansatz_circuit, 1)
        node2 = qnet.MeasureNode(1, 2, [0], ansatz_circuit, 1)

        ansatz_wires = qnet.NetworkAnsatz.collect_wires([node1, node2])
        assert ansatz_wires.tolist() == [1, 0]

        node3 = qnet.MeasureNode(1, 2, [0], ansatz_circuit, 1)
        with pytest.raises(
            ValueError,
            match="One or more wires are not unique. Each node must contain unique wires.",
        ):
            qnet.NetworkAnsatz.collect_wires([node2, node3])

    def test_network_settings(self):
        def ansatz_circuit(settings, wires):
            return None

        prep_nodes = [
            qnet.PrepareNode(3, [0], ansatz_circuit, 2),
            qnet.PrepareNode(2, [1], ansatz_circuit, 4),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], ansatz_circuit, 1),
            qnet.MeasureNode(1, 2, [1], ansatz_circuit, 3),
            qnet.MeasureNode(2, 2, [2, 3], ansatz_circuit, 2),
        ]
        network_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        zero_settings = network_ansatz.zero_network_settings()

        assert np.allclose(zero_settings, [0] * 23)
        assert isinstance(zero_settings, list)
        assert all([qml.math.requires_grad(setting) for setting in zero_settings])
        assert all([isinstance(setting, np.tensor) for setting in zero_settings])

        np.random.seed(123)
        rand_settings = network_ansatz.rand_network_settings()

        match_settings = [
            1.2344523,
            -1.34372619,
            -1.71624293,
            0.3224202,
            1.37896421,
            -0.48313636,
            3.02073055,
            1.1613195,
            -0.1198084,
            -0.67784562,
            -0.98534158,
            1.43916176,
            -0.38596197,
            -2.76662537,
            -0.64060684,
            1.49536924,
            -1.99496329,
            -2.03919676,
            0.19824313,
            0.19997863,
            0.84446613,
            2.19554471,
            1.4102944,
        ]

        assert np.allclose(rand_settings, match_settings)
        assert isinstance(rand_settings, list)
        assert all([qml.math.requires_grad(setting) for setting in rand_settings])
        assert all([isinstance(setting, np.tensor) for setting in rand_settings])

        # tensorflow types
        np.random.seed(123)
        tf_rand_settings = network_ansatz.tf_rand_network_settings()

        assert np.allclose(tf_rand_settings, match_settings)
        assert isinstance(tf_rand_settings, list)
        assert all([isinstance(setting, tf.Variable) for setting in tf_rand_settings])

    def test_fixed_network_settings(self):
        def ansatz_circuit(settings, wires):
            return None

        prep_nodes = [
            qnet.PrepareNode(2, [0], ansatz_circuit, 1),
            qnet.PrepareNode(2, [1], ansatz_circuit, 1),
        ]
        meas_nodes = [
            qnet.MeasureNode(2, 2, [0], ansatz_circuit, 1),
            qnet.MeasureNode(2, 2, [1], ansatz_circuit, 1),
        ]
        network_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

        np.random.seed(123)
        rand_settings = network_ansatz.rand_network_settings(
            fixed_setting_ids=[0, 2, 4, 6], fixed_settings=[0, 2, 4, 6]
        )

        match_settings = [
            0,
            -1.34372619,
            2,
            0.3224202,
            4,
            -0.48313636,
            6,
            1.1613195,
        ]

        assert np.allclose(rand_settings, match_settings)
        assert all([qml.math.requires_grad(rand_settings[i]) for i in [1, 3, 5, 7]])
        assert all([not (qml.math.requires_grad(rand_settings[i])) for i in [0, 2, 4, 6]])

        np.random.seed(123)
        tf_rand_settings = network_ansatz.tf_rand_network_settings(
            fixed_setting_ids=[0, 2, 4, 6], fixed_settings=[0, 2, 4, 6]
        )

        assert np.allclose(tf_rand_settings, match_settings)
        assert all([isinstance(tf_rand_settings[i], tf.Variable) for i in [1, 3, 5, 7]])
        assert all([isinstance(tf_rand_settings[i], tf.Tensor) for i in [0, 2, 4, 6]])
