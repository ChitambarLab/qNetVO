import pennylane as qml
from pennylane import math
import numpy as np
from ..qnodes import density_matrix_qnode
from ..utilities import partial_transpose


def negativity_cost_fn(network_ansatz, m, n, wires, qnode_kwargs={}):
    """Constructs an ansatz-specific negativity cost function.

    Negativity can be used to identify if two subsystems :math:`A` and :math:`B` are
    entangled, through the PPT criterion. Negativity is an upper bound for distillable entanglement.

    This entanglement measure is expressed as

    .. math::

            \\mathcal{N}(\\rho) = |\\sum_{\\lambda_i < 0}\\lambda_i|,

    where :math:`\\rho^{\\Gamma_B}` is the partial transpose of the joint state with respect to
    the :math:`B` party, and :math:`\\lambda_i` are all of the eigenvalues of :math:`\\rho^{\\Gamma_B}`.

    For more information on negativity and its applications in quantum information theory,
    (see `Vidal and Werner, 2001 <https://arxiv.org/pdf/quant-ph/0102117>`_).

    :param ansatz: The ansatz circuit on which the negativity is evaluated.
    :type ansatz: NetworkAnsatz

    :param m: The size of the :math:`A` subsystem.
    :type m: int

    :param n: The size of the :math:`B` subsystem.
    :type n: int

    :param wires: The wires which define the joint state.
    :type wires: list[int]

    :param qnode_kwargs: Keyword arguments passed to the execute qnodes.
    :type qnode_kwargs: dictionary

    :returns: A cost function ``negativity_cost(*network_settings)`` parameterized by
              the ansatz-specific scenario settings.
    :rtype: Function

    :raises ValueError: If the sum of the sizes of the two subsystems (``m + n``) does not match the length of ``wires``.
    """

    if len(wires) != m + n:
        raise ValueError(f"Sum of sizes of two subsystems should be {len(wires)}; got {m+n}.")

    density_qnode = density_matrix_qnode(network_ansatz, wires, **qnode_kwargs)

    def negativity_cost(*network_settings):
        dm = density_qnode(network_settings)
        dm_pt = partial_transpose(dm, 2**m, 2**n)
        eigenvalues = math.eigvalsh(dm_pt)
        negativity = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
        return -negativity

    return negativity_cost
