.. qnetvo documentation master file, created by
   sphinx-quickstart on Thu Jul 15 10:27:48 2021.

qNetVO: the Quantum Network Variational Optimizer
=================================================

*A Python package for simulating and optimizing quantum communication
networks on quantum hardware.*

Features
--------

Simulate Quantum Communication Networks on Quantum Computers:

* Create quantum circuits that model distributed quantum communication devices.
* Combine devices to flexibly construct quantum communication network simulations. 
* Execute ansatz circuits on quantum computing platforms and simulators.

Optimize Quantum Communication Networks using Variational Techniques:

* Train quantum communication networks like neural networks using differential programming.
* Use our library of communication oriented cost functions or create your own.

Powered by `PennyLane <https://pennylane.ai>`_:

* Free and open-source framework for cross-platform quantum machine learning.

Quick Start
-----------

.. todo::

   Register Package.

Install qNetVO:

.. code-block:: bash

   pip install qnetvo

Import qNetVO:

.. code-block:: python

   import qnetvo as qnet

How to Cite
-----------

.. todo::

   Register DOI for project.

.. todo::

   Add Citation.bib file to project.

Acknowledgments
---------------

We thank `Xanadu, Inc. <https://www.xanadu.ai/>`_, the
`UIUC Physics Department <https://physics.illinois.edu/>`_, and the
`Quantum Information Science and Engineering Network (QISE-Net)
<https://qisenet.uchicago.edu/>`_
for their support of this project.
Work funded by NSF award DMR-1747426.

License
-------

qNetVO is free and open-source.
The software is released under the Apache License, Version 2.0.
See `LICENSE <https://https://github.com/ChitambarLab/qNetVO/blob/main/LICENSE>`_ for details.

Site Navigation
---------------

Looking for something specific? See our :ref:`search` and :ref:`genindex`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   network_ansatzes/index
   cost/index
   optimization
   information
   utilities
