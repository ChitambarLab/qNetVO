.. QNetOptimizer documentation master file, created by
   sphinx-quickstart on Thu Jul 15 10:27:48 2021.

NISQNet.py
==========

*Simulate and optimize quantum prepare and measure networks using
noisy intermediate scale quantum (NISQ) computers.*

Features
========

Simulate Quantum Communication Networks on Quantum Computers:

* Create quantum circuits that model quantum communication devices.
* Combine devices to flexibly construct quantum communication network ansatzes. 
* Execute ansatz circuits on quantum computing platforms and simulators.

Optimimze Quantum Communication Networks:

* Apply differential programming to train quantum communication networks like neural networks.
* Use our library of communication oriented cost functions or create your own.

Powered by `PennyLane <https://pennylane.ai>`_:

* Free and open-source framework for cross-platform quantum machine learning.

Quick Start
===========

.. todo::

   Register Package.

Install NISQNet:

.. code-block:: bash

   pip install nisqnet

Import NISQNet:

.. code-block:: python

   import nisqnet as qnet

How to Cite
===========

.. todo::

   Register DOI for project.

.. todo::

   Add Citation.bib file to project.

Acknowledgments
===============

We thank `Xanadu, Inc. <https://www.xanadu.ai/>`_, the
`UIUC Physics Department <https://physics.illinois.edu/>`_, and the
`Quantum Information Science and Engineering Network (QISE-Net)
<https://qisenet.uchicago.edu/>`_
for their support of this project.
Work funded by NSF award DMR-1747426.

License
=======

.. todo::

   Add a project license.


Site Navigation
===============

Looking for something specific? See our :ref:`search` and :ref:`genindex`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   network_ansatzes/index
   cost/index
   optimization
   information
   utilities
