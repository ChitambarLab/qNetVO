The Magic Squares Game
======================

.. currentmodule:: qnetvo


Magic squares is a cooperative game where two players Alice and Bob each produce a
three-bit string :math:`(a_0,a_1,a_2)` and :math:`(b_0,b_1,b_2)`.
Alice's bitstring is required to have even parity while Bob's must have odd parity.
Alice's bitstring is then used to fill in the :math:`x^{th}` row of a 3x3 grid while
Bob's bitstring is used to fill in the :math:`y^{th}` column of a 3x3 grid.
The game is won if :math:`a_y = b_x` that is, the intesection of Alice's row and Bob's
column is filled in with same value by both Alice and Bob.

The magic square game is interesting because classically the winning probability is bounded
as :math:`P_{Win}\leq 8/9`.
On the other hand, there exists a quantum strategy that can win the game with certainty.
The foundational physics behind the winning strategy was independently observed by D. Mermin
[Mermin1990]_, and A. Peres [Peres1990]_.
Mermin and Peres' thought experiment in quantum noncontextuality was later showed to achieve a
perfect winning score in the magic square game by P. Aravind [Aravind2002]_, [Aravind2004]_.
In our implementation of the magic squares game, we apply the approach outlined by G. Brassard *et al.* [Brassard2005]_.
The optimal quantum strategy reduces to a bipartite non-signaling scenario where Alice and Bob perform local measurements on a shared static entangled state.
The optimal quantum protocol is performed as follows:

1. Alice and Bob receive independent classical inputs :math:`x,\;y\in \{0,1,2\}`.
2. Alice and Bob apply local unitaries :math:`U^A_x` and :math:`U^B_y` respectively.
3. Alice and Bob measure two-qubits in the computational basis to obtain two-bit outcomes
   :math:`(a_0,a_1)` and :math:`(b_0, b_1)` respectively where :math:`a_i,\;b_j \in \{0,1\}`.
4. Alice's third bit :math:`a_2` is selected such that the bitstring :math:`(a_0,a_1, a_2)` has
   even parity while Bob's third bit is selected such that the bitstring :math:`(b_0,b_1,b_2)`
   has odd parity.
5. The winning condition :math:`a_y = b_x` is evaluated.

As an example, optimal state preparation and measurements are provided in [Brassard2005]_.

.. autofunction:: magic_squares_game_cost

References
----------

.. [Mermin1990] Mermin, N. David. "Simple unified form for the major no-hidden-variables
   theorems." Physical review letters 65.27 (1990): 3373.

.. [Peres1990] Peres, Asher. "Incompatible results of quantum measurements." Physics Letters
   A 151.3-4 (1990): 107-108.

.. [Aravind2002] Aravind, Padmanabhan K. "Bell’s theorem without inequalities and only two
   distant observers." Journal of Genetic Counseling 15.4 (2002): 397-405.

.. [Aravind2004] Aravind, Padmanabhan K. "Quantum mysteries revisited again." American Journal
   of Physics 72.10 (2004): 1303-1307.

.. [Brassard2005] Brassard, Gilles, Anne Broadbent, and Alain Tapp. "Quantum pseudo-telepathy."
   Foundations of Physics 35.11 (2005): 1877-1907.