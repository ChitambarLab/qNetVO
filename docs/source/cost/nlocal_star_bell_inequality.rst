n-Local Star Bell Inequality
=============================

.. currentmodule:: qnetvo

The :math:`n`-local star network consists of :math:`n` entanglement sources (static preparation nodes) and :math:`n+1` measurement nodes where one central node holds one qubit from each source and the remaining :math:`n` nodes are connected to the central node through a single source.
The network forms an :math:`n` point star. 
Polynomial Bell inequalities can be derivied that tightly bound the set of classical correlations
[Tavakoli2014]_.
These :math:`n`-local star Bell inequalities witness quantum violations to the classical
:math:`n`-local set and can be used to optimize non-:math:`n`-locality in star configurations.


.. autofunction:: nlocal_star_22_cost_fn

.. autofunction:: star_I22_fn

.. autofunction:: star_J22_fn

.. autofunction:: parallel_nlocal_star_grad_fn

References
----------

.. [Tavakoli2014] Tavakoli, Armin, et al. "Nonlocal correlations in the star-network
   configuration." Physical Review A 90.6 (2014): 062109.

