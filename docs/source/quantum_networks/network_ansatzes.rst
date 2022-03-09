Network Ansatzes
================

.. currentmodule:: qnetvo

The quantum network ansatz is a collection of network nodes aggregated
into three sequential layers: prepare, noise, and measure.
The nodes in each layer are independent and each operate upon a unique set of wires.
If a prepaare and measure node each operate on the same wire, then quantum communicaation
is sent from the prepare node to the measure node.

.. autoclass:: NetworkAnsatz
	:members: