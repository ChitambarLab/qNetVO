from functools import wraps
import pennylane as qml


class PrepareNode:
    def __init__(self, num_in, wires, quantum_fn, num_settings):
        self.num_in = num_in
        self.wires = wires
        self.ansatz_fn = quantum_fn
        self.num_settings = num_settings

    def settings_dims(self):
        return (self.num_in, self.num_settings)


class MeasureNode:
    def __init__(self, num_in, num_out, wires, quantum_fn, num_settings):
        self.num_in = num_in
        self.num_out = num_out
        self.wires = wires
        self.ansatz_fn = quantum_fn
        self.num_settings = num_settings

    def settings_dims(self):
        return (self.num_in, self.num_settings)


def prepare_layer(prepare_nodes):
    def prepare_circuit(settings_array):

        num_nodes = len(prepare_nodes)

        for node_id in range(num_nodes):
            node = prepare_nodes[node_id]
            node.ansatz_fn(settings_array[node_id], node.wires)

    return prepare_circuit


def measure_layer(measure_nodes):
    def measure_circuit(settings_array):

        num_nodes = len(measure_nodes)

        for node_id in range(num_nodes):
            node = measure_nodes[node_id]
            node.ansatz_fn(settings_array[node_id], node.wires)

    return measure_circuit


class NetworkAnsatz:
    def __init__(self, prepare_nodes, measure_nodes):
        self.prepare_nodes = prepare_nodes
        self.measure_nodes = measure_nodes

        self.prepare_wires = self.collect_wires(prepare_nodes)
        self.measure_wires = self.collect_wires(measure_nodes)
        self.network_wires = qml.wires.Wires.all_wires([self.prepare_wires, self.measure_wires])

        self.dev = qml.device("default.qubit", wires=self.network_wires)
        self.fn = self.construct_ansatz_circuit()

    def construct_ansatz_circuit(self):
        prep_layer = prepare_layer(self.prepare_nodes)
        meas_layer = measure_layer(self.measure_nodes)

        def ansatz_circuit(prepare_settings_array, measure_settings_array):
            prep_layer(prepare_settings_array)
            meas_layer(measure_settings_array)

        return ansatz_circuit

    @staticmethod
    def collect_wires(network_nodes):
        ansatz_wires = list(map(lambda node: qml.wires.Wires(node.wires), network_nodes))
        all_wires = qml.wires.Wires.all_wires(ansatz_wires)
        unique_wires = qml.wires.Wires.unique_wires(ansatz_wires)

        # two nodes cannot prepare a state on the same wire
        if not all_wires.tolist() == unique_wires.tolist():
            raise ValueError(
                "One or more wires are not unique. Each node must contain unique wires."
            )

        return all_wires
