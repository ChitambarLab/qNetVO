import pytest

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
