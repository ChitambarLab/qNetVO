

I-3322 Inequality
=================

.. currentmodule:: qnetvo


Consider the bipartite non-signaling scenario where Alice and Bob each have three classical
inputs and two classical outputs.
This scenario is modeled as a prepare and measure network with one preparation node
(entanglement source) and two measurement nodes (Alice and Bob).
The local Hilbert space dimensions used by Alice and Bob depend on the particular ansatz
and are specified by the number of wires in each measurement node.

The Bell inequalities bounding this scenario are computed in full by I. Pitowsky and K. Svozil [Pitowsky2001]_.
It was later shown by D. Collins and N. Gisin [Collins2004]_ that there is one novel Bell inequality for this scenario, the :math:`I_{3322}` inequality.
For inputs :math:`x,y\in \{0,1,2\}` and outputs :math:`a,b\in \{0,1\}`, the :math:`I_{3322}` Bell inequality is expressed as

.. math::

   I_{3322} := &-P_A(0|0) -2P_B(0|0) - P_B(0|1) + P(00|00) +P(00|01) + P(00|02) \\
               &+ P(00|10) + P(00|11) - p(00|12) + P(00|20) - P(00|21) 


where :math:`P(a,b|x,y)` are conditional joint probalities and :math:`P_A` (:math:`P_B`)
are local probabilities for Alice (Bob).
The classical upper bound is known to be :math:`I_{3322} \leq \beta_C = 0` while the quantum
upper bound for local qubit Hilbert spaces is :math:`I_{3322} \leq \beta_Q = 0.25` and has been
found to be bounded by :math:`\beta_Q \leq 0.250\: 875\: 38` for local Hilbert spaces
infinite dimension [Pal2010]_.


.. autofunction:: post_process_I_3322_joint_probs

.. autofunction:: I_3322_bell_inequality_cost_fn


References
----------


.. [Pitowsky2001] Pitowsky, Itamar, and Karl Svozil. "Optimal tests of quantum nonlocality."
   Physical Review A 64.1 (2001): 014102.

.. [Collins2004] Collins, Daniel, and Nicolas Gisin. "A relevant two qubit Bell inequality
   inequivalent to the CHSH inequality." Journal of Physics A: Mathematical and General 37.5
   (2004): 1775.

.. [Pal2010] Pál, Károly F., and Tamás Vértesi. "Maximal violation of a bipartite three-setting,
   two-outcome Bell inequality using infinite-dimensional quantum systems." Physical Review A
   82.2 (2010): 022116.
