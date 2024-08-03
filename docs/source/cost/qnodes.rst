QNodes
======

.. currentmodule:: qnetvo


A :meth:`qnetvo.NetworkAnsatz` is executed on a `PennyLane QNode <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QNode.html?highlight=qnode#qml-qnode>`_ object.
The output of the QNode execution can either be the probility of obtaining
each output or the expectation of an observable.
The considered cost function dictates which type of QNode measurement is applied.

Probability QNodes
------------------

.. autofunction:: joint_probs_qnode

Density Matrix QNodes
---------------------

.. autofunction:: density_matrix_qnode

Parity Observable QNodes
------------------------

.. autofunction:: local_parity_expval_qnode

.. autofunction:: global_parity_expval_qnode

Helper Functions for Parity Observables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: parity_observable

.. autofunction:: local_parity_observables

.. autofunction:: parity_vector

.. autofunction:: even_parity_ids

