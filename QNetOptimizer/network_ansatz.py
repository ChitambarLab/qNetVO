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
