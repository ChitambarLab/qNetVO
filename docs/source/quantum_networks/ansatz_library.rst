Ansatz Library
==============

.. currentmodule:: qnetvo

For convenience a library of state preparations, unitaries, and
circuit-based noise model are provided.

State Preparations
------------------

.. autofunction:: bell_state_copies

.. autofunction:: ghz_state

.. autofunction:: max_entangled_state

.. autofunction:: nonmax_entangled_state

.. autofunction:: graph_state_fn

.. autofunction:: W_state


Unitary Layers
--------------

.. autofunction:: local_RY

.. autofunction:: local_RXRY

.. autofunction:: local_Rot


Noise Models
------------

.. autofunction:: pure_amplitude_damping

.. autofunction:: pure_phase_damping

.. autofunction:: two_qubit_depolarizing

.. autofunction:: colored_noise