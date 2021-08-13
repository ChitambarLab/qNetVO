from pennylane import numpy as np


def parity_vector(n_qubits):
    """Constructs a vector with elements :math:`\\pm 1` describing the parity of
    an :math:`n`-bit string measured from :math:`n`-qubits.
    The (+) and (-) elements indicate even and odd parity respectively and are placed in
    the index corresponding to the bit string's value.

    :param n_qubits: The number of qubits for which to consider the parity vector.
    :type n_qubits: int

    :raises ValueError: If ``n_qubits < 1``.
    """
    if n_qubits < 1:
        raise ValueError("Input `n_qubits` must satisfy `n_qubits >= 1`.")

    return (
        np.array([1, -1], dtype=int)
        if n_qubits == 1
        else np.kron([1, -1], parity_vector(n_qubits - 1))
    )


def even_parity_ids(n_qubits):
    """Constructs the list ids corresponding to even parity bit strings with respect
    to the vector returned by ``parity_vector``.

    :param n_qubits: The number of qubits for which to consider the parity vector.
    :type n_qubits: int
    """
    p_vec = parity_vector(n_qubits)
    return np.argwhere(p_vec == 1).flatten()
