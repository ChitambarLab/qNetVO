.. qnetvo documentation master file, created by
   sphinx-quickstart on Thu Jul 15 10:27:48 2021.

qNetVO: the Quantum Network Variational Optimizer
=================================================

*Simulate and optimize quantum communication networks using quantum computers.*

Features
--------

QNetVO simulates quantum communication networks on differentiable quantum cicuits.
The cicuit parameters are optimized with respect to a cost function using gradient descent.
qNetVO is powered by `PennyLane <https://pennylane.ai>`_ an open-source framework
for cross-platform quantum machine learning.

Simulating Quantum Communication Networks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Construct complex quantum network ansatzes from generic quantum circuit compenents.
* Simulate the quantum network on a quantum computer or classical simulator.

Optimizing Quantum Communication Networks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Use our library of network oriented cost functions or create your own.
* Gradient descent methods for tuning quantum network ansatz settings to minimize the cost.

Quick Start
-----------

Install qNetVO:

.. code-block::

   $ pip install qnetvo

Install PennyLane:

.. code-block::

   $ pip install pennylane==0.20

Import packages:

.. code-block:: python

   import pennylane as qml
   import qnetvo as qnet

.. note::

   For optimal use, qNetVO should be used with PennyLane.
   QNetVO is currently compatible with PennyLane v0.20.

Site Navigation
---------------

Looking for something specific? See our :ref:`search` and :ref:`genindex`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quantum_networks/index
   cost/index
   optimization
   utilities

How to Cite
-----------

.. todo::

   Register DOI for project.

.. todo::

   Add Citation.bib file to project.

License
-------

qNetVO is free and open-source.
The software is released under the Apache License, Version 2.0.
See `LICENSE <https://https://github.com/ChitambarLab/qNetVO/blob/main/LICENSE>`_ for details.

Acknowledgments
---------------

We thank `Xanadu, Inc. <https://www.xanadu.ai/>`_, the
`UIUC Physics Department <https://physics.illinois.edu/>`_, and the
`Quantum Information Science and Engineering Network (QISE-Net)
<https://qisenet.uchicago.edu/>`_
for their support of this project.
Work funded by NSF award DMR-1747426.
