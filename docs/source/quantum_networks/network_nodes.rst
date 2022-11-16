Network Nodes
=============

.. currentmodule:: qnetvo

A quantum network is constructed from a collection of quantum devices serving
as nodes in the network.
A network node performs a quantum operation on its local set of qubits.
The applied operation is conditioned upon a classical input. 

Network nodes are categorized by their function:

* **Prepare Node:** Initializes a quantum state on the local wires. 
* **Processing Node:** Applies a quantum operation on its local wires.
* **Noise Node:** Applies a static "noisy" operation to the local wires.
* **Measure Node:** Performs a measurement operation on the local wires and outputs a classical value.

Prepare Nodes
-------------

.. autoclass:: PrepareNode
	:members:

Processing Nodes
-------------

.. autoclass:: ProcessingNode
	:members:

Noise Nodes
-----------

.. autoclass:: NoiseNode
    :members:

Measure Nodes
-------------

.. autoclass:: MeasureNode
	:members:
