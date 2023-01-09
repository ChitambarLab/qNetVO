import pytest
import pennylane as qml
from pennylane import numpy as np

import tensorflow as tf

import qnetvo as qnetvo


@pytest.fixture
def chsh_ansatz():
    prepare_nodes = [qnetvo.PrepareNode(1, [0, 1], qnetvo.local_RY, 2)]
    measure_nodes = [
        qnetvo.MeasureNode(2, 2, [0], qnetvo.local_RY, 1),
        qnetvo.MeasureNode(2, 2, [1], qnetvo.local_RY, 1),
    ]

    return qnetvo.NetworkAnsatz(prepare_nodes, measure_nodes)


@pytest.fixture
def ex_network_ansatz():
    circuit = lambda settings, wires: qml.RY(settings[0], wires=wires[0])
    noise = lambda settings, wires: qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])
    cc_meas = lambda settings, wires: [qml.measure(wires[0])]
    cond_circ = lambda settings, wires, classical_wires: qml.cond(classical_wires[0], qml.PauliX)(
        wires[0]
    )

    prepare_nodes = [
        qnetvo.PrepareNode(1, [0], circuit, 1),
        qnetvo.PrepareNode(1, [1], circuit, 1),
        qnetvo.PrepareNode(1, [2], circuit, 1),
    ]
    cc_sender_nodes = [qnetvo.CCSenderNode(2, [3], [0], cc_meas, 0)]
    noise_nodes = [
        qnetvo.NoiseNode([1], noise),
        qnetvo.NoiseNode([2], noise),
    ]
    cc_receiver_nodes = [qnetvo.CCReceiverNode(1, [0], [0], cond_circ, 0)]
    measure_nodes = [
        qnetvo.MeasureNode(3, 2, [0], circuit, 1),
        qnetvo.MeasureNode(2, 2, [1], circuit, 1),
    ]

    return qnetvo.NetworkAnsatz(
        prepare_nodes,
        cc_sender_nodes,
        noise_nodes,
        cc_receiver_nodes,
        measure_nodes,
        dev_kwargs={"name": "default.mixed"},
    )


def test_network_ansatz_init_attributes(ex_network_ansatz):
    # verify network layers
    assert len(ex_network_ansatz.layers) == 5
    assert all([isinstance(node, qnetvo.PrepareNode) for node in ex_network_ansatz.layers[0]])
    assert all([isinstance(node, qnetvo.CCSenderNode) for node in ex_network_ansatz.layers[1]])
    assert all([isinstance(node, qnetvo.NoiseNode) for node in ex_network_ansatz.layers[2]])
    assert all([isinstance(node, qnetvo.CCReceiverNode) for node in ex_network_ansatz.layers[3]])
    assert all([isinstance(node, qnetvo.MeasureNode) for node in ex_network_ansatz.layers[4]])

    assert ex_network_ansatz.layers_num_settings == [3, 0, 0, 0, 2]
    assert ex_network_ansatz.layers_total_num_in == [1, 2, 1, 1, 6]

    assert ex_network_ansatz.layers_node_num_in == [[1, 1, 1], [2], [1, 1], [1], [3, 2]]
    assert ex_network_ansatz.layers_num_nodes == [3, 1, 2, 1, 2]

    # verify network wires
    assert len(ex_network_ansatz.layers_wires) == 5
    assert ex_network_ansatz.layers_wires[0].tolist() == [0, 1, 2]
    assert ex_network_ansatz.layers_wires[1].tolist() == [3]
    assert ex_network_ansatz.layers_wires[2].tolist() == [1, 2]
    assert ex_network_ansatz.layers_wires[3].tolist() == [0]
    assert ex_network_ansatz.layers_wires[4].tolist() == [0, 1]

    assert ex_network_ansatz.network_wires.tolist() == [0, 1, 2, 3]

    # verify cc_wires
    assert ex_network_ansatz.network_cc_wires.tolist() == [0]
    assert ex_network_ansatz.num_cc_wires == 1

    assert ex_network_ansatz.layers_cc_wires_in[0].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_in[1].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_in[2].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_in[3].tolist() == [0]
    assert ex_network_ansatz.layers_cc_wires_in[4].tolist() == []

    assert ex_network_ansatz.layers_cc_wires_out[0].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_out[1].tolist() == [0]
    assert ex_network_ansatz.layers_cc_wires_out[2].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_out[3].tolist() == []
    assert ex_network_ansatz.layers_cc_wires_out[4].tolist() == []

    # verify parameter partitions

    print(ex_network_ansatz.parameter_partitions)
    assert ex_network_ansatz.parameter_partitions == [
        [[(0, 1)], [(1, 2)], [(2, 3)]],
        [[(3, 3), (3, 3)]],
        [[(3, 3)], [(3, 3)]],
        [[(3, 3)]],
        [[(3, 4), (4, 5), (5, 6)], [(6, 7), (7, 8)]],
    ]

    # verify device
    assert ex_network_ansatz.dev_kwargs["name"] == "default.mixed"
    assert ex_network_ansatz.dev.wires.tolist() == [0, 1, 2, 3]
    assert ex_network_ansatz.dev.short_name == "default.mixed"

    # verify qnode construction and execution
    @qml.qnode(ex_network_ansatz.dev)
    def noisy_test_circuit(settings):
        ex_network_ansatz.fn(settings)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    assert np.isclose(noisy_test_circuit([0, 0, 0, 0, 0]), 0.5)
    assert np.isclose(noisy_test_circuit([np.pi / 4, -np.pi / 3, 0, -np.pi / 4, np.pi / 3]), 0.5)


def test_init_device_default(chsh_ansatz):
    assert chsh_ansatz.dev.short_name == "default.qubit"


def test_network_ansatz_set_device(chsh_ansatz):
    assert chsh_ansatz.dev.short_name == "default.qubit"
    assert chsh_ansatz.dev_kwargs["name"] == "default.qubit"

    updated_dev = chsh_ansatz.set_device("default.mixed", shots=5)
    assert updated_dev == chsh_ansatz.dev
    assert chsh_ansatz.dev.short_name == "default.mixed"
    assert chsh_ansatz.dev_kwargs["name"] == "default.mixed"
    assert chsh_ansatz.dev.shots == 5
    assert chsh_ansatz.dev_kwargs["shots"] == 5


def test_network_ansatz_device_method(chsh_ansatz):
    dev1 = chsh_ansatz.device()
    dev2 = chsh_ansatz.device()

    assert dev1 != dev2
    assert dev1.short_name == dev2.short_name
    assert dev1.shots == dev2.shots


def test_partition_settings_slices():
    prep_nodes = [
        qnetvo.PrepareNode(3, [0, 1], qml.ArbitraryStatePreparation, 6),
        qnetvo.PrepareNode(2, [2], qnetvo.local_RY, 1),
    ]
    meas_nodes = [
        qnetvo.MeasureNode(2, 2, [1], qml.ArbitraryUnitary, 3),
        qnetvo.MeasureNode(2, 4, [0, 2], qml.ArbitraryUnitary, 15),
    ]

    network_ansatz = qnetvo.NetworkAnsatz(prep_nodes, meas_nodes)

    assert network_ansatz.get_network_parameter_partitions() == [
        [[(0, 6), (6, 12), (12, 18)], [(18, 19), (19, 20)]],
        [[(20, 23), (23, 26)], [(26, 41), (41, 56)]],
    ]


def test_qnode_settings(chsh_ansatz):
    np.random.seed(123)
    network_settings = chsh_ansatz.rand_network_settings()

    settings = chsh_ansatz.qnode_settings(network_settings, [[0], [0, 1]])

    assert np.allclose(settings, [1.2344523, -1.34372619, -1.71624293, -0.48313636])


@pytest.mark.parametrize(
    "layer_inputs, layer_id, match",
    [([1, 0], 0, [1, 2]), ([0, 1], 0, [0, 3]), ([1], 1, [6, 7])],
)
def test_layer_settings(layer_inputs, layer_id, match):
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
        qnetvo.PrepareNode(2, [0], qnetvo.local_RY, 1),
        qnetvo.PrepareNode(2, [1], qnetvo.local_RY, 1),
    ]
    meas_nodes = [qnetvo.MeasureNode(2, 2, [0, 1], qnetvo.local_RY, 2)]
    network_ansatz = qnetvo.NetworkAnsatz(prep_nodes, meas_nodes)

    settings = network_ansatz.layer_settings(network_settings, layer_id, layer_inputs)
    assert np.allclose(settings, match)


def test_circuit_layer():
    def ansatz_circuit(settings, wires):
        qml.RY(settings[0], wires=wires[0])

    def noisy_ansatz_circuit(settings, wires):
        qml.Hadamard(wires=wires[0])
        qml.DepolarizingChannel(0.5 * 3 / 4, wires=wires[0])

    node1 = qnetvo.PrepareNode(1, [0], ansatz_circuit, 1)
    node2 = qnetvo.PrepareNode(1, [1], ansatz_circuit, 1)

    noisy_node1 = qnetvo.NoiseNode([0], noisy_ansatz_circuit)
    noisy_node2 = qnetvo.NoiseNode([1], noisy_ansatz_circuit)

    @qml.qnode(qml.device("default.qubit", wires=2))
    def test_circuit(settings):
        qnetvo.NetworkAnsatz.circuit_layer_fn([node1, node2])(settings, cc_wires=[])

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    assert np.isclose(test_circuit([0, 0]), 1)
    assert np.isclose(test_circuit([np.pi, 0]), -1)

    val = test_circuit([np.pi / 4, -np.pi / 4])
    assert np.isclose(val, 0.5)

    @qml.qnode(qml.device("default.mixed", wires=2))
    def noisy_test_circuit(settings):
        qnetvo.NetworkAnsatz.circuit_layer_fn([noisy_node1, noisy_node2])(settings, cc_wires=[])

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


@pytest.mark.parametrize(
    "wires_lists, check_unique, match",
    [
        ([[0], [1]], True, [0, 1]),
        ([[0, 3], [2, 4]], True, [0, 3, 2, 4]),
        ([[0, 1], [1, 2], [3, 2]], False, [0, 1, 2, 3]),
        ([[], [], []], True, []),
        ([[0], [], [1]], True, [0, 1]),
    ],
)
def test_collect_wires(wires_lists, check_unique, match):
    wires = qnetvo.NetworkAnsatz.collect_wires(wires_lists, check_unique)
    assert isinstance(wires, qml.wires.Wires)
    assert wires.tolist() == match


@pytest.mark.parametrize(
    "wires_lists",
    [
        [[0, 1], [1, 2], [3, 2]],
        [[0], [0], [1, 2, 3, 4]],
    ],
)
def test_collect_wires_check_unique_failures(wires_lists):
    with pytest.raises(
        ValueError,
        match="One or more wires are not unique. Each node must contain unique wires.",
    ):
        qnetvo.NetworkAnsatz.collect_wires(wires_lists)


@pytest.mark.parametrize(
    "cc_wires_in_layers, cc_wires_out_layers",
    [
        ([[], [0, 1], [2, 3]], [[0, 1, 2], [3], []]),
        ([[], [], [], []], [[], [], [], []]),
        ([[], [], [0, 1, 2, 3]], [[0, 3], [1, 2], [4]]),
    ],
)
def test_check_cc_causal_structure_success(cc_wires_in_layers, cc_wires_out_layers):
    assert qnetvo.NetworkAnsatz.check_cc_causal_structure(
        list(map(qml.wires.Wires, cc_wires_in_layers)),
        list(map(qml.wires.Wires, cc_wires_out_layers)),
    )


@pytest.mark.parametrize(
    "cc_wires_in_layers, cc_wires_out_layers, i",
    [
        ([[0], [1]], [[0, 1], []], 0),
        ([[], [1], [2], [4]], [[1], [2, 3], []], 3),
        ([[], [0, 1]], [[0], [1]], 1),
    ],
)
def test_check_cc_causal_structure_errors(cc_wires_in_layers, cc_wires_out_layers, i):
    with pytest.raises(
        ValueError,
        match="The `cc_wires_in` of layer "
        + str(i)
        + " do not have corresponding `cc_wires_out` in a preceding layer.",
    ):
        qnetvo.NetworkAnsatz.check_cc_causal_structure(
            list(map(qml.wires.Wires, cc_wires_in_layers)),
            list(map(qml.wires.Wires, cc_wires_out_layers)),
        )


def test_network_settings():
    def ansatz_circuit(settings, wires):
        return None

    prep_nodes = [
        qnetvo.PrepareNode(3, [0], ansatz_circuit, 2),
        qnetvo.PrepareNode(2, [1], ansatz_circuit, 4),
    ]
    meas_nodes = [
        qnetvo.MeasureNode(2, 2, [0], ansatz_circuit, 1),
        qnetvo.MeasureNode(1, 2, [1], ansatz_circuit, 3),
        qnetvo.MeasureNode(2, 2, [2, 3], ansatz_circuit, 2),
    ]
    network_ansatz = qnetvo.NetworkAnsatz(prep_nodes, meas_nodes)

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


def test_fixed_network_settings():
    def ansatz_circuit(settings, wires):
        return None

    prep_nodes = [
        qnetvo.PrepareNode(2, [0], ansatz_circuit, 1),
        qnetvo.PrepareNode(2, [1], ansatz_circuit, 1),
    ]
    meas_nodes = [
        qnetvo.MeasureNode(2, 2, [0], ansatz_circuit, 1),
        qnetvo.MeasureNode(2, 2, [1], ansatz_circuit, 1),
    ]
    network_ansatz = qnetvo.NetworkAnsatz(prep_nodes, meas_nodes)

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


@pytest.mark.parametrize(
    "network_input, match_settings",
    [
        ([[0], [0, 0]], [1, 2, 3, 0, 4, 0]),
        ([[0], [0, 1]], [1, 2, 3, 0, 0, 4]),
        ([[0], [1, 0]], [1, 2, 0, 3, 4, 0]),
        ([[0], [1, 1]], [1, 2, 0, 3, 0, 4]),
    ],
)
def test_expand_qnode_settings_chsh_ansatz(chsh_ansatz, network_input, match_settings):
    qnode_settings = np.array([1, 2, 3, 4])
    network_settings = chsh_ansatz.expand_qnode_settings(qnode_settings, network_input)

    assert all(network_settings == match_settings)
