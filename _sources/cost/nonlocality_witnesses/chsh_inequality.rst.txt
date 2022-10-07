CHSH Inequality
===============

.. currentmodule:: qnetvo

The CHSH scenario consists of two nonsignaling devices which share randomness in the
classical case and entanglement in the quantum case.

.. figure:: ../../_static/images/bipartite_non-signaling.png
   :scale: 60%
    
   **Bipartite Non-Signaling Scenario.** 

The set of classical behaviors is bound by the CHSH inequality [Clauser1969]_

.. math::

	I_{CHSH} := \left \vert\sum_{x,y=0}^1 (-1)^{x\cdot y}\langle A_xB_y \rangle\right \vert \leq 2,

where :math:`\langle A_xB_y \rangle = \text{Tr}[(A_x\otimes B_y )\rho^{AB}]` is a bipartite
correlator for dichotomic observables :math:`A_x` and :math:`B_y` with eigenvalues :math:`\pm  1`.
Quantum entanglement yields a maximal CHSH score of of :math:`I_{CHSH}=\leq \beta^Q_{CHSH} = 2\sqrt{2}`
[Cirelson1980]_.


.. autofunction:: chsh_inequality_cost_fn

.. autofunction:: parallel_chsh_grad_fn


References
----------

.. [Clauser1969] Clauser, John F., et al. "Proposed experiment to test local hidden-variable theories."
   Physical review letters 23.15 (1969): 880.

.. [Cirelson1980] Cirel'son, Boris S. "Quantum generalizations of Bell's inequality."
   Letters in Mathematical Physics 4.2 (1980): 93-100.