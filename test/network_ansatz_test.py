import pytest
import numpy as np
import pennylane as qml

from context import QNetOptimizer as QNopt


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
        assert prep_node.settings_dims() == (3, 2)


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
        assert measure_node.settings_dims() == (3, 2)


class TestPrepareLayer:
    def test_basic_use(self):
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        node1 = QNopt.PrepareNode(1, [0], ansatz_circuit, 1)
        node2 = QNopt.PrepareNode(1, [1], ansatz_circuit, 1)

        dev = qml.device("default.qubit", wires=[0, 1])

        @qml.qnode(dev)
        def test_circuit(settings):
            QNopt.prepare_layer([node1, node2])(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert test_circuit([[0], [0]]) == 1
        assert test_circuit([[np.pi], [0]]) == -1

        val = test_circuit([[np.pi / 4], [-np.pi / 4]])
        assert np.isclose(val, 0.5)


class TestMeasureLayer:
    def test_basic_use(self):
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        node1 = QNopt.MeasureNode(1, 2, [0], ansatz_circuit, 1)
        node2 = QNopt.MeasureNode(1, 2, [1], ansatz_circuit, 1)

        dev = qml.device("default.qubit", wires=[0, 1])

        @qml.qnode(dev)
        def test_circuit(settings):
            QNopt.measure_layer([node1, node2])(settings)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert test_circuit([[0], [0]]) == 1
        assert test_circuit([[np.pi], [0]]) == -1

        val = test_circuit([[np.pi / 4], [-np.pi / 4]])
        assert np.isclose(val, 0.5)


class TestNetworkAnsatz:
    def test_init(self):
        # setup test
        def ansatz_circuit(settings, wires):
            qml.RY(settings[0], wires=wires[0])

        prepare_nodes = [
            QNopt.PrepareNode(1, [0], ansatz_circuit, 1),
            QNopt.PrepareNode(1, [1], ansatz_circuit, 1),
            QNopt.PrepareNode(1, [2], ansatz_circuit, 1),
        ]
        measure_nodes = [
            QNopt.MeasureNode(1, 2, [0], ansatz_circuit, 1),
            QNopt.MeasureNode(1, 2, [1], ansatz_circuit, 1),
        ]

        network_ansatz = QNopt.NetworkAnsatz(prepare_nodes, measure_nodes)

        # verify network nodes
        assert network_ansatz.prepare_nodes == prepare_nodes
        assert network_ansatz.measure_nodes == measure_nodes

        # verify wires
        assert network_ansatz.prepare_wires.tolist() == [0, 1, 2]
        assert network_ansatz.measure_wires.tolist() == [0, 1]
        assert network_ansatz.network_wires.tolist() == [0, 1, 2]

        # verify device
        assert network_ansatz.dev.wires.tolist() == [0, 1, 2]
        assert network_ansatz.dev.short_name == "default.qubit"

        # verify qnode construction and execution
        @qml.qnode(network_ansatz.dev)
        def test_circuit(prepare_settings_array, measure_settings_array):
            network_ansatz.fn(prepare_settings_array, measure_settings_array)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert test_circuit([[0], [0], [0]], [[0], [0]]) == 1
        assert test_circuit([[np.pi / 4], [-np.pi / 3], [0]], [[-np.pi / 4], [np.pi / 3]]) == 1

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
