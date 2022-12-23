import pytest
import pennylane as qml

import qnetvo


def test_network_node():
    def circuit(settings, wires, cc_wires):
        qml.cond(cc_wires[0], qml.ArbitraryUnitary)(settings, wires)

    node = qnetvo.NetworkNode(
        num_in=2,
        wires=[0, 1],
        ansatz_fn=circuit,
        num_settings=15,
        cc_wires_in=[4],
    )

    assert isinstance(node, qnetvo.NetworkNode)

    assert node.num_in == 2
    assert node.wires == [0, 1]
    assert node.ansatz_fn == circuit
    assert node.num_settings == 15
    assert node.cc_wires_in == [4]

    mock_measurement = qml.measurements.MeasurementValue(["12345678"], lambda v: v)
    mock_settings = list(range(15))

    with qml.tape.QuantumTape() as tape:
        node(mock_settings, [mock_measurement])

    assert len(tape) == 1
    assert tape.wires.tolist() == [0, 1]
    assert tape.num_params == 0
    assert tape.get_parameters() == []


def test_noise_node():
    def circuit(settings, wires):
        qml.AmplitudeDamping(0.7, wires[0])
        qml.AmplitudeDamping(0.3, wires[1])

    noise_node = qnetvo.NoiseNode([0, 1], circuit)

    assert noise_node.wires == [0, 1]
    assert noise_node.ansatz_fn == circuit
    assert noise_node.num_in == 1
    assert noise_node.num_settings == 0
    assert noise_node.cc_wires_in == []

    with qml.tape.QuantumTape() as tape:
        noise_node([])

    assert len(tape) == 2
    assert tape.wires.tolist() == [0, 1]
    assert tape.num_params == 2
    assert tape.get_parameters() == [0.7, 0.3]


def test_processing_node():
    def circuit(settings, wires):
        qml.RY(settings[0], wires=wires[0])
        qml.RZ(settings[1], wires=wires[1])

    proc_node = qnetvo.ProcessingNode(3, [0, 1], circuit, 2)

    assert isinstance(proc_node, qnetvo.ProcessingNode)

    assert proc_node.num_in == 3
    assert proc_node.wires == [0, 1]
    assert proc_node.ansatz_fn == circuit
    assert proc_node.num_settings == 2
    assert proc_node.cc_wires_in == []

    with qml.tape.QuantumTape() as tape:
        proc_node([0.5, 0.6])

    assert len(tape) == 2
    assert tape.wires.tolist() == [0, 1]
    assert tape.num_params == 2
    assert tape.get_parameters() == [0.5, 0.6]


def test_prepare_node():
    def circuit(settings, wires):
        qml.ArbitraryStatePreparation(settings, wires=wires[0:2])

    prep_node = qnetvo.PrepareNode(4, [2, 3], circuit, 6)

    assert isinstance(prep_node, qnetvo.PrepareNode)

    assert prep_node.num_in == 4
    assert prep_node.wires == [2, 3]
    assert prep_node.ansatz_fn == circuit
    assert prep_node.num_settings == 6
    assert prep_node.cc_wires_in == []

    with qml.tape.QuantumTape() as tape:
        prep_node([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    assert len(tape) == 1
    assert tape.wires.tolist() == [2, 3]
    assert tape.num_params == 1
    assert tape.get_parameters() == [[0, 0.1, 0.2, 0.3, 0.4, 0.5]]


def test_measure_node():
    def circuit(settings, wires=[0, 1]):
        qml.RY(settings[0], wires=wires[0])
        qml.RZ(settings[1], wires=wires[1])

    meas_node = qnetvo.MeasureNode(3, 4, [0, 1], circuit, 2)

    assert isinstance(meas_node, qnetvo.MeasureNode)

    assert meas_node.num_in == 3
    assert meas_node.num_out == 4
    assert meas_node.wires == [0, 1]
    assert meas_node.ansatz_fn == circuit
    assert meas_node.num_settings == 2
    assert meas_node.cc_wires_in == []

    with qml.tape.QuantumTape() as tape:
        meas_node([0.1, 0.5])

    assert len(tape) == 2
    assert tape.wires.tolist() == [0, 1]
    assert tape.num_params == 2
    assert tape.get_parameters() == [0.1, 0.5]


def test_cc_measure_node():
    def circuit(settings, wires):
        qml.CNOT(wires[0:2])
        qml.Hadamard(wires[0])

        bit_0 = qml.measure(wires[0])
        bit_1 = qml.measure(wires[1])

        return [bit_0, bit_1]

    cc_meas_node = qnetvo.CCMeasureNode(
        num_in=1, wires=[0, 1], cc_wires_out=[1, 2], ansatz_fn=circuit, num_settings=0
    )

    assert isinstance(cc_meas_node, qnetvo.CCMeasureNode)

    assert cc_meas_node.num_in == 1
    assert cc_meas_node.wires == [0, 1]
    assert cc_meas_node.cc_wires_out == [1, 2]
    assert cc_meas_node.ansatz_fn == circuit
    assert cc_meas_node.num_settings == 0
    assert cc_meas_node.cc_wires_in == []

    with qml.tape.QuantumTape() as tape:
        cc_meas_node([])

    assert len(tape) == 4
    assert tape.wires.tolist() == [0, 1]
    assert tape.num_params == 0
    assert tape.get_parameters() == []


# class TestNoiseNode:
#     def test_init(self):
#         def circuit(settings, wires=[0, 1]):
#             qml.AmplitudeDamping(0.7, wires=[0])
#             qml.AmplitudeDamping(0.3, wires=[1])

#         noise_node = qnet.NoiseNode([0, 1], circuit)

#         assert noise_node.wires == [0, 1]
#         assert noise_node.ansatz_fn == circuit
#         assert noise_node.num_in == 1
#         assert noise_node.num_settings == 0


# @pytest.fixture
# def base_node_kwargs():
#     return {
#         "num_in":2,
#         "wires":[0,1],
#         "ansatz_fn" : lambda settings, wires, cc_wires: qml.cond(cc_wires[0], qml.ArbitraryUnitary(*settings, wires=wires)),
#         "num_settings" : 15,
#         "cc_wires_in" : [4],
#     }

# @pytest.fixture
# def base_node2_kwargs():
#     return {
#         "num_in": 1,
#         "wires": [0],
#         "ansatz_fn" : lambda settings, wires: qml.Rot(*settings, wires),
#         "num_settings": 3,
#         "cc_wires_in":[],
#     }

# @pytest.fixture
# def base_node(base_node_kwargs):
#     return qnetvo.NetworkNode(**base_node_kwargs)

# @pytest.fixture
# def base_node2(base_node2_kwargs):
#     return qnetvo.NetworkNode(**base_node2_kwargs)


# @pytest.mark.parametrize(
#     "node_class, node_kwargs",
#     [
#         (base_node_kwargs, base_node),
#         (base_node2_kwargs, base_node2),
#     ],
#     indirect=True
# )
# def test_network_node(node_class, node_kwargs):

#     assert isinstance(node, qnetvo.NetworkNode)

#     assert node.num_in == node_kwargs["num_in"]
