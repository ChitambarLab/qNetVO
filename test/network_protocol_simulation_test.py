import pytest
import pennylane as qml

import numpy as np

import qnetvo


@pytest.fixture
def teleportation_ansatz():
    prep_nodes = [
        qnetvo.PrepareNode(
            wires=[0],
            ansatz_fn=qml.ArbitraryStatePreparation,
            num_settings=2,
        ),
        qnetvo.PrepareNode(
            wires=[1, 2],
            ansatz_fn=qnetvo.ghz_state,
        ),
    ]

    def cc_send_circ(settings, wires):
        qml.adjoint(qnetvo.ghz_state)(settings, wires[0:2])

        b0 = qml.measure(wires[0])
        b1 = qml.measure(wires[1])

        return [b0, b1]

    cc_sender_nodes = [
        qnetvo.CCSenderNode(
            wires=[0, 1],
            cc_wires_out=[0, 1],
            ansatz_fn=cc_send_circ,
        )
    ]

    def meas_circ(settings, wires, cc_wires):
        qml.cond(cc_wires[0], qml.PauliZ)(wires[0])
        qml.cond(cc_wires[1], qml.PauliX)(wires[0])

    cc_receiver_nodes = [qnetvo.CCReceiverNode(wires=[2], ansatz_fn=meas_circ, cc_wires_in=[0, 1])]

    return qnetvo.NetworkAnsatz(prep_nodes, cc_sender_nodes, cc_receiver_nodes)


@pytest.mark.parametrize(
    "settings, match_vec",
    [
        ([0, 0], np.array([1, 0])),
        ([np.pi / 2, 0], np.array([1, -1j]) / np.sqrt(2)),
        ([np.pi, 0], np.array([0, 1])),
        ([0, np.pi / 2], np.array([1, 1]) / np.sqrt(2)),
        ([0, np.pi], np.array([0, 1])),
        ([np.pi, np.pi / 2], np.array([1, -1]) / np.sqrt(2)),
    ],
)
def test_teleportation(teleportation_ansatz, settings, match_vec):
    @qml.qnode(qml.device(**teleportation_ansatz.dev_kwargs))
    def teleport(settings):
        teleportation_ansatz(settings)
        return qml.density_matrix(wires=[2])

    rho = teleport(settings)
    rho_target = np.outer(match_vec, match_vec.conj())

    assert np.allclose(rho, rho_target)


def test_shared_randomness():
    def cc_meas_circ(settings, wires):
        qml.Hadamard(wires[0])
        return [qml.measure(wires[0])]

    cc_sender_nodes = [qnetvo.CCSenderNode(wires=[0], cc_wires_out=[0], ansatz_fn=cc_meas_circ)]

    def shared_random_circ(settings, wires, cc_wires):
        qml.cond(cc_wires[0], qml.PauliX)(wires[0])
        qml.cond(cc_wires[0], qml.PauliX)(wires[1])

    cc_receiver_nodes = [
        qnetvo.CCReceiverNode(wires=[1, 2], ansatz_fn=shared_random_circ, cc_wires_in=[0])
    ]

    ansatz = qnetvo.NetworkAnsatz(cc_sender_nodes, cc_receiver_nodes)

    @qml.qnode(qml.device(**ansatz.dev_kwargs))
    def circ():
        ansatz()
        return qml.density_matrix([1, 2])

    assert np.allclose(
        circ(), np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]) / 2
    )


def test_superdense_coding():
    prep_nodes = [qnetvo.PrepareNode(wires=[0, 1], ansatz_fn=qnetvo.ghz_state)]

    def superdense_encode_circ(settings, wires):
        qml.PhaseShift(settings[0], wires[0])
        qml.RY(settings[1], wires[0])

        qml.PhaseShift(settings[2], wires[0])

    proc_nodes = [
        qnetvo.ProcessingNode(num_in=4, wires=[0], ansatz_fn=superdense_encode_circ, num_settings=3)
    ]

    meas_nodes = [
        qnetvo.MeasureNode(num_out=4, wires=[0, 1], ansatz_fn=qml.adjoint(qnetvo.ghz_state))
    ]

    sd_coding_ansatz = qnetvo.NetworkAnsatz(prep_nodes, proc_nodes, meas_nodes)

    @qml.qnode(qml.device(**sd_coding_ansatz.dev_kwargs))
    def superdense_coding(input):
        network_settings = [0, 0, 0, np.pi, np.pi, 0, 0, 0, np.pi, np.pi, np.pi, np.pi]
        qn_settings = sd_coding_ansatz.qnode_settings(network_settings, [[], [input], []])

        sd_coding_ansatz(qn_settings)

        return qml.probs(wires=[0, 1])

    assert np.allclose(superdense_coding(0), [1, 0, 0, 0])
    assert np.allclose(superdense_coding(1), [0, 1, 0, 0])
    assert np.allclose(superdense_coding(2), [0, 0, 1, 0])
    assert np.allclose(superdense_coding(3), [0, 0, 0, 1])
