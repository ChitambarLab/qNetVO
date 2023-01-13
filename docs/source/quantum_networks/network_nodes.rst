Network Nodes
=============

.. currentmodule:: qnetvo

A quantum network is constructed from a collection of quantum devices serving
as nodes in the network.
Each network node operates on its local set of qubits.
Each node's operation can be conditioned upon both classical inputs and
the measurement results of other nodes in the network. 

.. autoclass:: NetworkNode
	:members:

Network nodes are categorized by their function:

* **Prepare Node:** Initializes a quantum state on the local qubits. 
* **Processing Node:** Applies an operation to its local qubits.
* **Noise Node:** Applies noise to the local qubits.
* **Measure Node:** Measures local qubits and outputs a classical value.
* **CC Sender Node:** Measures local qubits and sends the result to classical communication (CC) receiver nodes.
* **CC Receiver Node:** Applies an operation to its local qubits conditioned upon receieved classical data.

Prepare Nodes
-------------

.. autoclass:: PrepareNode
	:members:

Processing Nodes
----------------

.. autoclass:: ProcessingNode
	:members:

Measure Nodes
-------------

.. autoclass:: MeasureNode
	:members:


Classical Communication Nodes
-----------------------------

.. autoclass:: CCSenderNode
	:members:

.. autoclass:: CCReceiverNode
	:members:


Noise Nodes
-----------

.. autoclass:: NoiseNode
    :members: