Network Nodes
=============

.. currentmodule:: qnetvo

A quantum network is constructed from a collection of quantum devices serving
as nodes in the network.
A network node performs a quantum opteration on a local set of a wires.
In some cases, nodes can be given a classical input on which to condition the applied
operation and/or a classical output obtained after the applying the quantum operation. 

Netwok nodes are categorized by their function:

* **Prepare Node:** Initializes a quantum state on the local wires.
* **Processing Node:** Applies an operation on its local wires 
* **Noise Node:** Applies a static "noisy" operation to the local wires.
* **Measure Node:** Performs a measurement operation on the local wires.

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
