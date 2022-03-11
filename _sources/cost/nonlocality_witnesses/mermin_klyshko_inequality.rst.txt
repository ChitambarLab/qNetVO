Mermin-Klyshko Inequality
=========================

.. currentmodule:: qnetvo

Consider a non-signaling scenario with a single source and :math:`n` measurement
devices.
The Mermin-Klyshko (MK) inequality bounds the classical correlations in such scenarios
but yields a quantum violation that increases exponentially with :math:`n` [Mermin1990]_,
[Ardehali1992]_, [Belinsky1993_1]_, [Belinsky1993_2]_. 
The inequality is expressed iteratively as [Cabello2002]_,

.. math::

    I_{\text{MK}} := M_n = M_{n-1}(B^n_0 + B^n_1) + M'_{n-1}(B^n_0 - B^n_1) \leq 2^{n-1}

where :math:`B^j_{y_j}` is the dichotomic observable for the :math:`j^{th}` measurement
node and :math:`y_j\in\{0,1\}`.
Additionally, the term :math:`M'_j` is derived from :math:`M_j` simply by exchanging the
subscripts :math:`0 \leftrightarrow 1` describing the measurement input at each node.
The inequality construction is initialized with :math:`M_1 = B^1_0` and :math:`M'_1 = B^1_1`.
The inequality :math:`I_{\text{MK}}` contains :math:`2^{2 \lfloor n/2 \rfloor}` correlator
terms resulting in an exponential computational complexity in the number of quantum circuits
to execute.

The classical bound for the form of the inequality :math:`I_{\text{MK}}` is :math:`2^{n-1}`
while the quantum bound is :math:`2^{3(n-1)/2}` [Cabello2002]_.
A remarkable feature of the Mermin-Klyshko inequality is that the 
separation between the quantum and classical bounds grows exponentialy with :math:`n`.


.. autofunction:: mermin_klyshko_cost_fn

.. autofunction:: mermin_klyshko_inputs_scalars

.. autofunction:: mermin_klyshko_classical_bound

.. autofunction:: mermin_klyshko_quantum_bound

References
----------

.. [Mermin1990] Mermin, N. David. "Extreme quantum entanglement in a superposition of 
   macroscopically distinct states." Physical Review Letters 65.15 (1990): 1838.

.. [Ardehali1992] Ardehali, Mohammad. "Bell inequalities with a magnitude of violation that grows
   exponentially with the number of particles." Physical Review A 46.9 (1992): 5375.

.. [Belinsky1993_1] Belinsky, A. V., and D. N. Klyshko. "A modified N-particle Bell theorem, the
   corresponding optical experiment and its classical model." Physics Letters A 176.6 (1993): 
   415-420.

.. [Belinsky1993_2] Belinski, A. V., and David Nikolaevich Klyshko. "Interference of light and
   Bell's theorem." Physics-Uspekhi 36.8 (1993): 653.

.. [Cabello2002] Cabello, Adan. "Bell’s inequality for n spin-s particles." Physical Review A
   65.6 (2002): 062105.