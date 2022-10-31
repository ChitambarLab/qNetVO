Network Ansatzes
================

.. currentmodule:: qnetvo

The quantum network ansatz is a collection of network nodes aggregated
into sequential layers ordered as prepare nodes, noise or processing nodes, and measure nodes.
It is important that all measurement nodes are in the final layer.
The nodes in each layer are independent and each operate upon a unique set of wires.
If a two sequential nodes operate on the same wire, then one-way quantum communication
is sent from the first node to the second node.

.. autoclass:: NetworkAnsatz
	:members: